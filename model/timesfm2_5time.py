import timesfm
import math
import numpy as np
import torch

class Model:
    def __init__(self, model_path):
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_path, 
            local_files_only=True
        )
        self.model.compile(timesfm.ForecastConfig(
            max_context=128,
            max_horizon=128, 
            normalize_inputs=True,
        ))
        self.period = 24

    def forecast(self, pred_len, inputs, args=None):
        """
        inputs: [Batch, seq_Len] (Tensor)
        outputs: [Batch, pred_len] (Tensor)
        """
        B, seq_len = inputs.shape
        
        remainder = seq_len % self.period
        if remainder != 0:
            inputs = inputs[:, remainder:]
            seq_len = inputs.shape[1]
            
        num_days = seq_len // self.period
        
        # [Batch, Days, 24]
        x_reshaped = inputs.contiguous().view(B, num_days, self.period)
        # -> [Batch, 24, Days] -> [Batch * 24, Days]
        inputs_daily = x_reshaped.permute(0, 2, 1).reshape(B * self.period, num_days)
        pred_steps = math.ceil(pred_len / self.period)
        # 如果原本预测120点, 这里变成5点
        outputs, quant= self.model.forecast(horizon = pred_steps, inputs = inputs_daily)
        outputs = quant[:, :, args.quant]
        # TimesFM 返回的是 numpy array, 转回 Tensor
        if isinstance(outputs, np.ndarray):
            outputs  = torch.from_numpy(outputs)
        outputs = outputs.to(inputs.device)
        
        # [Batch*24, Steps] -> [Batch, 24, Steps]
        forecast_reshaped = outputs.view(B, self.period, pred_steps)
        
        # -> [Batch, Steps, 24] -> (Day1_0-23h, Day2_0-23h...)
        forecast_ordered = forecast_reshaped.permute(0, 1, 2).transpose(1, 2)
        
        # 展平 [Batch, Steps * 24]
        final_output = forecast_ordered.contiguous().reshape(B, pred_steps * self.period)
        
        return final_output