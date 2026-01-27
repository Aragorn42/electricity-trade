import torch
from transformers import AutoModel

class Model():
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            'ant-intl/Falcon-TST_Large',
            trust_remote_code=True
        )

    def forecast(self, pred_len, inputs, args):
        predictions = self.model.predict(inputs, forecast_horizon=pred_len)
        return predictions
