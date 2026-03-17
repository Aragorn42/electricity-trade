import timesfm
import math
import numpy as np
import torch
from chronos import Chronos2Pipeline
import torch
import math
import numpy as np
from einops import rearrange
class Model:
    def __init__(self, args):
        # 假设 Chronos2Pipeline 已经正确导入
        self.model = Chronos2Pipeline.from_pretrained(
            "/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Chronos2/", 
            device_map="cuda"
        )
        self.default_quant = args.quant  # 保留默认quant作为fallback
        self.period = 24

    def forecast(self, pred_len, inputs, args, quants=None):
        """
        inputs: [Batch, seq_Len] (Tensor)
        quants: list or array of length 24 (Optional). 
                If provided, contains the quantile index for each hour (0-23).
                If None, uses args.quant for all hours.
        outputs: [Batch, pred_len] (Tensor)
        """
        B, seq_len = inputs.shape
        num_days = seq_len // self.period

        x_reshaped = inputs.contiguous().view(B, num_days, self.period)
        
        pred_steps = math.ceil(pred_len / self.period)
        hourly_results = []
        for h in range(self.period):
            input_h = x_reshaped[:, :, h].unsqueeze(1) # -> [Batch, 1, Days]
            pred_df = self.model.predict(
                input_h,
                prediction_length=pred_steps
            )
            
            full_forecast = torch.cat(pred_df, dim=0) # [Batch, Num_Quantiles, pred_steps]
            # output_h shape -> [Batch, Num_Quantiles, pred_steps]
            output_h = full_forecast
            if isinstance(output_h, np.ndarray):
                output_h = torch.from_numpy(output_h)
            output_h = output_h.to(inputs.device)
            
            hourly_results.append(output_h)
        # stack 之后 -> [Batch, 24, Num_Quantiles, pred_steps]
        forecast_stacked = torch.stack(hourly_results, dim=1)

        return rearrange(forecast_stacked, 'b h q p -> b (h p) q')