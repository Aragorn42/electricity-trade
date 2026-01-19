import pandas as pd
from chronos import Chronos2Pipeline
from chinese_calendar import is_holiday
import torch

class Model():
    def __init__(self, args):
        self.model = Chronos2Pipeline.from_pretrained("/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Chronos2/", device_map="cuda")
        self.quant = args.quant

    def forecast(self, args, x, holiday_x, holiday_y):
        batch_size = x.shape[0]
        inputs_list = []
        for i in range(batch_size):
            item_dict = {
                "target": x[i], 
                "past_covariates": {
                    "is_holiday": holiday_x[i]
                },
                "future_covariates": {
                    "is_holiday": holiday_y[i]
                }
            }
            inputs_list.append(item_dict)
        batch_forecasts = self.model.predict(
            inputs=inputs_list,
            prediction_length=args.pred_len,
            batch_size=batch_size 
        )
        batch_preds_tensor = torch.stack(batch_forecasts)
        preds = batch_preds_tensor[:, 0, self.quant, :]
        return preds