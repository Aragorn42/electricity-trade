import os

import torch
from transformers import AutoModelForCausalLM

# ==========================================
# 1. 自动配置 HF 镜像 (放在 import transformers 之前)
# ==========================================


class Model():
    def __init__(self, args):
        print("正在从 hf-mirror.com 加载 Time-MoE 模型...")
        
        # 使用 args 中的路径，或者默认使用官方 ID
        # 如果你已经下载到了本地，把 model_path 改成本地路径即可
        model_path = "Maple728/TimeMoE-200M" 
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,  # 自动分配到 GPU
            trust_remote_code=True   # Time-MoE 必须开启此选项
        )
        # Time-MoE 是点预测模型，通常不需要像 Chronos 那样选分位数
        # 但保留这个变量以兼容你的接口

        self.quant = args.quant 

        def skip_validation(model_kwargs):
            return 
    
        self.model._validate_model_kwargs = skip_validation

    def forecast(self, pred_len, inputs, args):
        """
        inputs: tensor, shape usually (batch_size, context_len)
        """
        # 确保输入在正确的设备上
        past_values = inputs.to(self.device)
        
        # Time-MoE 的输入维度通常是 (Batch, Seq_Len)
        # 你的原始代码 inputs 似乎已经是这个维度，不需要像 Chronos 那样 unsqueeze(1)
        # 如果 inputs 是 3D (Batch, 1, Seq_Len)，则需要 squeeze 掉中间那维
        if past_values.dim() == 3:
            past_values = past_values.squeeze(1)

        with torch.no_grad():
            # 调用 Time-MoE 的 generate 方法
            # 注意：Time-MoE 的 generate 接口参数可能随版本更新
            # 通常只需要 past_values 和 prediction_length
            forecast_result = self.model.generate(past_values,
                max_new_tokens=pred_len,
                # 如果模型支持 output_hidden_states 等参数可在此添加
            )
        #print(forecast_result.shape)
        # forecast_result 的 shape 通常是: (batch_size, pred_len)
        # 或者是 (batch_size, pred_len, dim)，根据模型具体输出调整

        return forecast_result.cpu() # 转回 CPU 以匹配后续处理