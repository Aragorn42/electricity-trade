import timesfm
import torch
import math
import numpy as np

class Model():
    def __init__(self, model_path):
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_path, 
            local_files_only=True
        )
        self.model.compile(timesfm.ForecastConfig(
            max_context=64,
            max_horizon=32, 
            normalize_inputs=True,
            use_continuous_quantile_head=True, # 概率预测
            force_flip_invariance=True, # 输入-X保证输出-Y
            infer_is_positive=False, # 强制输出非负
            fix_quantile_crossing=False # 分位数保序
        ))

    def forecast(self, pred_len, inputs, args, quants=None):
        """
        基于 self.model.forecast(pred_len, inputs) 接口实现
        inputs: [Batch, seq_Len]
        """
        B, seq_len = inputs.shape
        period = 24
        num_days = seq_len // period
        pred_steps = math.ceil(pred_len / period)
        
        # 1. 定义分位数索引策略
        if quants is None:
            quants = [-1] * 24
        
        # 2. 重塑输入数据: [Batch, Days, 24]
        x_reshaped = inputs[:, :num_days * period].contiguous().view(B, num_days, period)
        
        hourly_results = []
        
        # 3. 按小时(0-23)独立进行日频预测
        for h in range(period):
            # 提取第 h 小时所有历史天的数据 -> [Batch, num_days]
            input_h = x_reshaped[:, :, h]
            
            # 调用模型接口预测未来 pred_steps 天
            # 返回的 y_pred_h 形状为 [Batch, pred_steps, num_quantiles]
            _, y_pred_h = self.model.forecast(pred_steps, input_h)
            if isinstance(y_pred_h, np.ndarray):
                y_pred_h = torch.from_numpy(y_pred_h).to(inputs.device)
            # 根据策略提取该小时特定的分位数
            current_quant_idx = quants[h]
            output_h = y_pred_h[:, :, current_quant_idx] # [Batch, pred_steps]
            
            hourly_results.append(output_h)
            
        # 4. 交叉合并结果
        # stack -> [Batch, 24, pred_steps]
        forecast_stacked = torch.stack(hourly_results, dim=1)
        
        # 置换维度还原时间轴顺序: [Batch, DayStep, Hour]
        forecast_ordered = forecast_stacked.permute(0, 2, 1) 
        
        # 展平为连续的小时序列: [Batch, pred_steps * 24]
        final_output = forecast_ordered.contiguous().view(B, -1)
        
        # 5. 截断多余部分并返回
        return final_output[:, :pred_len]