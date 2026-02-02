import torch
from torch import nn
from transformers import AutoModelForCausalLM

class Model(nn.Module):
    def __init__(self):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True)
    def forecast(self, pred_len, inputs, args):
        # tensor
        pred_df = self.model.generate(
            inputs, max_new_tokens=120
        )
        return pred_df.squeeze(1)
