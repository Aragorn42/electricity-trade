from transformers import AutoModelForCausalLM
import torch

class Model():
    def __init__(self, args):
        self.model = AutoModelForCausalLM.from_pretrained('/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Yinglong', trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    
    def forecast(self, pred_len, inputs, args):
        inputs = inputs.to(torch.device("cuda"))
        y_pred = self.model.generate(inputs, future_token=pred_len)
        y_pred = y_pred[:, :, args.quant]
        return y_pred.detach().cpu().float().numpy()
