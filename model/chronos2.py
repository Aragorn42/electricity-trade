import pandas as pd
from chronos import Chronos2Pipeline
import torch
class Model():
    def __init__(self, args):
        self.model = Chronos2Pipeline.from_pretrained("/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Chronos2/", device_map="cuda")
        self.quant = args.quant
    def forecast(self, pred_len, inputs, args):
        # tensor
        input3d = inputs.unsqueeze(1)
        pred_df = self.model.predict(
            input3d,
            prediction_length=pred_len
        )
        # pred_df : [batch_size, 1, probility, pred_len]
        full_forecast = torch.cat(pred_df, dim=0)
        # 注意：你的样本数 21 在 dim=1，所以要在 dim=1 上取中位数
        # 结果 shape: (64, pred_len)
        median_forecast = full_forecast[:, self.quant, :] 

        return median_forecast
