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
        
        x_reshaped = inputs[:, :num_days * period].contiguous().view(B, num_days, period)
        
        hourly_results = []
        # Batch, Days, 24]
        for h in range(period):
            # 取出当前小时的 Tensor [Batch, num_days]
            input_h_tensor = x_reshaped[:, :, h]
            input_h_numpy = input_h_tensor.detach().cpu().numpy()
            input_h_list = list(input_h_numpy)
   
            _, y_pred_h = self.model.forecast(pred_steps, input_h_list)
            
            if isinstance(y_pred_h, np.ndarray):
                y_pred_h = torch.from_numpy(y_pred_h).to(inputs.device)
            
            output_h = y_pred_h # [Batch, pred_steps, Num_Quantiles]
            hourly_results.append(output_h)

        forecast_stacked = torch.stack(hourly_results, dim=1) # [Batch, 24, pred_steps, Num_Quantiles]

        return forecast_stacked.view(B, -1, forecast_stacked.shape[-1])