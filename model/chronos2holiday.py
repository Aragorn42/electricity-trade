import pandas as pd
from chronos import Chronos2Pipeline
from chinese_calendar import is_holiday
import torch

class Model():
    def __init__(self, args):
        self.model = Chronos2Pipeline.from_pretrained("/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Chronos2/", device_map="cuda")
        self.quant = args.quant

    def forecast(self, args, x_da, x_rt, x, holiday_x, holiday_y):
        '''
        x_da: [batch, Seq]
        x_rt: [batch, Seq]
        x: [batch, Seq]
        holiday_x: [batch, Seq]
        holiday_y: [batch, Pred]
        '''
        batch_size = x.shape[0]
        inputs_list = []
        
        for i in range(batch_size):
            item_dict = {
                #"target": torch.cat([x_da[i], x_rt[i], x[i]], dim=0).view(3, -1), #  
                "past_covariates": {
                    "is_holiday": holiday_x[i]
                    #"da": x_da[i],
                    #"rt": x_rt[i],
                },
                "future_covariates": {
                    "is_holiday": holiday_y[i]
                },
                "target": x[i],
            }
            inputs_list.append(item_dict)
        batch_forecasts = self.model.predict(
            inputs=inputs_list,
            prediction_length=args.pred_len,
            batch_size=batch_size 
        )
        batch_preds_tensor = torch.stack(batch_forecasts)
        preds = batch_preds_tensor[:, -1, self.quant, :]
        return preds