import os

import torch
from transformers import AutoModelForCausalLM

class Model():
    def __init__(self, args):
        model_path = "Maple728/TimeMoE-200M" 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            trust_remote_code=True
        )
        self.quant = args.quant 
        def skip_validation(model_kwargs):
            return 
        self.model._validate_model_kwargs = skip_validation

    def forecast(self, pred_len, inputs, args):
        """
        inputs: tensor, shape usually (batch_size, context_len)
        """
        with torch.no_grad():
            forecast_result = self.model.generate(inputs,
                max_new_tokens=pred_len,
            )
            
        return forecast_result.cpu()