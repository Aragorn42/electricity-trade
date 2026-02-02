import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.dataset.common import ListDataset
from huggingface_hub import hf_hub_download

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

import torch
import numpy as np
import math
from gluonts.dataset.common import ListDataset

MODEL = "moirai2"  # model name: choose from {'moirai', 'moirai-moe', 'moirai2'}
SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 120  # prediction length: any positive integer
CTX = 200  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer

# Read data into pandas DataFrame
url = (
    "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
    "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
)
df = pd.read_csv(url, index_col=0, parse_dates=True)

# Convert into GluonTS dataset
ds = PandasDataset(dict(df))

# Split into train/test set
train, test_template = split(
    ds, offset=-TEST
)  # assign last TEST time steps as test set

# Construct rolling window evaluation
test_data = test_template.generate_instances(
    prediction_length=PDT,  # number of time steps for each prediction
    windows=TEST // PDT,  # number of windows in rolling window evaluation
    distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
)
class Model():
    def __init__(self):
        # if MODEL == "moirai":
        #     self.model = MoiraiForecast(
        #         module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
        #         prediction_length=PDT,
        #         context_length=CTX,
        #         patch_size=PSZ,
        #         num_samples=100,
        #         target_dim=1,
        #         feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        #         past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        #     )
        # elif MODEL == "moirai-moe":
        #     self.model = MoiraiMoEForecast(
        #         module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
        #         prediction_length=PDT,
        #         context_length=CTX,
        #         patch_size=16,
        #         num_samples=100,
        #         target_dim=1,
        #         feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        #         past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        #     )
        # elif MODEL == "moirai2":

        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(
                f"Salesforce/moirai-2.0-R-small",
            ),
            prediction_length=120,
            context_length=1680,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

    def forecast(self, pred_len, inputs, args, quants=None):
        """
        策略：
        1. 将输入 reshape 为 [Batch, Days, 24]。
        2. 循环 24 次（0-23时），每次取出一个 slice [Batch, Days, 1] 作为独立的日频序列。
        3. 对每个小时的序列预测 pred_steps (即 pred_len / 24) 步。
        4. 对每个小时应用特定的分位数。
        5. 拼接结果并还原时间顺序。
        """
        # 1. 维度准备
        B, seq_len = inputs.shape
        period = 24
        num_days = seq_len // period
        
        # 计算每个小时需要向后预测多少“天” (例如总长120小时 -> 预测5天)
        pred_steps = math.ceil(pred_len / period)
        
        # 2. 数据重塑: [Batch, seq_len] -> [Batch, num_days, 24]
        # 注意：这里需要先转 numpy 处理 ListDataset，同时保留 tensor 用于 device 引用
        inputs_np = inputs.detach().cpu().numpy()
        # 确保能整除，如果不能整除需要截断或填充，这里假设输入是完整的整天
        x_reshaped = inputs_np[:, :num_days*period].reshape(B, num_days, period)
        
        # 3. 分位数设置
        if quants is None:
            # 默认分位数配置 (参考你的代码)
            quants = [11, 13, 16, 16, 12, 14, 13, 13, 9, 15, 13, 14, 16, 13, 12, 17, 15, 12, 14, 14, 14, 17, 15, 13]
            
        # 假设 index 10 对应 0.5 (中位数)，步长 0.05
        def get_quantile_float(idx):
            return max(0.01, min(idx * 0.05, 0.99))

        # 4. 创建预测器
        # 注意：Moirai/GluonTS 的 predictor 通常在创建时绑定 prediction_length。
        # 这里我们需要预测 pred_steps (比如 5)，而不是原来的 pred_len (比如 120)。
        # 如果 create_predictor 支持参数，最好传入；如果不支持，则需要在取结果时截断。
        try:
            predictor = self.model.create_predictor(batch_size=BSZ, prediction_length=pred_steps)
        except TypeError:
            # 如果接口不支持 prediction_length 参数，就用默认的，后面再切片
            predictor = self.model.create_predictor(batch_size=BSZ)

        hourly_results = []
        
        # 5. 按小时循环 (h: 0 -> 23)
        for h in range(period):
            # 取出所有样本在当前小时 h 的历史数据 -> [Batch, num_days]
            # 这被视为一个频率为"D"(天)的时间序列
            input_h = x_reshaped[:, :, h]
            
            # 构造 GluonTS 数据集
            # start 设为 2024-01-01 00:00 并不影响相对预测，只要频率 freq="D" 正确
            ds_h = ListDataset(
                [
                    {
                        "target": row, 
                        "start": "2024-01-01 00:00:00" 
                    } 
                    for row in input_h
                ],
                freq="D" # 关键：这里必须是 Day，因为序列点之间间隔 24 小时
            )
            
            # 进行预测
            forecasts = list(predictor.predict(ds_h))
            
            # 提取当前小时对应的分位数
            current_quant_idx = quants[h]
            q_float = get_quantile_float(current_quant_idx)
            
            # 解析预测结果
            batch_h_preds = []
            for f in forecasts:
                # f.quantile(q) 返回 shape: (prediction_length_of_model, )
                # 我们只需要前 pred_steps 步
                val = f.quantile(q_float)
                
                # 截断以防模型输出比 pred_steps 长
                val = val[:pred_steps]
                
                # 转 tensor
                batch_h_preds.append(torch.tensor(val, dtype=torch.float32))
            
            # stack -> [Batch, pred_steps]
            # 代表所有 batch 在 Hour h 的未来 pred_steps 天的预测值
            output_h = torch.stack(batch_h_preds, dim=0).to(inputs.device)
            
            hourly_results.append(output_h)

        # 6. 拼接与重排
        # 此时 hourly_results 是一个 list，包含 24 个 tensor，每个形状 [Batch, pred_steps]
        
        # Stack -> [Batch, 24, pred_steps]
        forecast_stacked = torch.stack(hourly_results, dim=1)
        
        # 目标顺序: Day1_H0, Day1_H1... Day1_H23, Day2_H0...
        # 当前维度: [Batch, Hour(0-23), DayStep(0-4)]
        # 需要转置为: [Batch, DayStep, Hour]
        forecast_ordered = forecast_stacked.permute(0, 2, 1) 
        
        # 展平 -> [Batch, pred_steps * 24]
        # 例如: [Batch, 5 * 24] = [Batch, 120]
        final_output = forecast_ordered.contiguous().reshape(B, pred_steps * period)
        
        # 如果原始要求的 pred_len 不是 24 的倍数 (比如 100)，需要最后截断一下
        if final_output.shape[1] > pred_len:
            final_output = final_output[:, :pred_len]
            
        return final_output