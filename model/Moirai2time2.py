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

    def forecast(self, pred_len, inputs, args):
            inputs_np = inputs.detach().cpu().numpy()
            ds = ListDataset(
                [
                    {
                        "target": row,
                        "start": "2024-01-01 00:00" 
                    }
                    for row in inputs_np
                ],
                freq="H",
            )
            
            predictor = self.model.create_predictor(batch_size=BSZ)
            forecasts = list(predictor.predict(ds))
            
            # 策略定义
            period = 24
            #quant_indices = [5, 12, 12, 12, 9, 9, 14, 9, 12, 12, 10, 12, 13, 10, 9, 9, 8, 10, 12, 11, 14, 11, 11, 11]

            quant_indices = [0.7] * 24  # 使用中位数预测
            batch_preds = []
            for f in forecasts:
                # f.samples shape: (num_samples, prediction_length)
                length = min(pred_len, f.prediction_length)
                
                row_results = []
                for t in range(length):
                    hour_in_day = t % period
                    q_idx = quant_indices[hour_in_day]
                    q_float = q_idx
                    
                    # --- 修改开始 ---
                    # f.quantile(q) 返回的是一个长度为 prediction_length 的 numpy 数组
                    # 我们只需要当前时刻 t 的那个值
                    full_seq_quantile = f.quantile(q_float) 
                    val = full_seq_quantile[t] 
                    # --- 修改结束 ---
                    
                    row_results.append(val)
                
                # 此时 row_results 是一个全是标量(float)的列表
                # tensor 转换后 shape 为 (pred_len,)
                batch_preds.append(torch.tensor(row_results, dtype=torch.float32))
            
            # 堆叠后 shape: [Batch, pred_len]
            preds = torch.stack(batch_preds, dim=0)
            preds = preds.to(inputs.device)
            
            return preds