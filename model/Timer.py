import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM

class Model():
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True)
    def forecast(self, pred_len, inputs, args):
        # tensor
        pred_df = self.model.generate(
            inputs, max_new_tokens=120
        )
        return pred_df