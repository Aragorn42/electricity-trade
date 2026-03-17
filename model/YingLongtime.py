import torch
import math
import numpy as np
from transformers import AutoModelForCausalLM
import torch

class Model():
    def __init__(self, args):
        self.model = AutoModelForCausalLM.from_pretrained('/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Yinglong', trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    
    def forecast(self, pred_len, inputs, args, quants=None):
        # inputs: [Batch, seq_Len]
        return self.model.generate(inputs.to(torch.device("cuda")), future_token=pred_len).detach().cpu().float().numpy()# [Batch, pred_len, Quantiles]