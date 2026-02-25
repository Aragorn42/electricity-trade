import torch
import math
import numpy as np
from transformers import AutoModelForCausalLM
import torch

class Model():
    def __init__(self, args):
        self.model = AutoModelForCausalLM.from_pretrained('/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Yinglong', trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    
    def forecast(self, pred_len, inputs, args, quants):
        """
        inputs: [Batch, seq_Len]
        策略: 
        1. 模型一次性预测所有时间步的所有分位数。
        2. 使用 torch.gather 或 索引掩码，根据时间步 t 对应的 Hour，
           从最后一维(分位数维度)中提取出特定的值。
        """
        # 1. 设备与模型推理
        device = torch.device("cuda")
        inputs = inputs.to(device)
        #print(quants)
        #quants = [0, 72, 85, 85, 94, 80, 88, 90, 90, 95, 68, 82, 90, 74, 85, 80, 80, 72, 84, 74, 80, 78, 95, 75]
        # full_pred 包含了所有可能的分位数结果
        full_pred = self.model.generate(inputs, future_token=pred_len)
        # 2. 定义分位数策略 (Hour 0-23 对应的 quantile index)
        indices_list = [quants[t % 24] for t in range(pred_len)]
        
        # 转为 Tensor: [pred_len]
        indices_tensor = torch.tensor(indices_list, device=device, dtype=torch.long)
        
        # 4. 使用 gather 提取特定分位数
        # full_pred shape: [Batch, pred_len, Quantiles]
        # 我们需要构建一个与 full_pred 前两维匹配的索引张量来进行 gather
        
        batch_size = full_pred.shape[0]
        
        # 变形索引: [pred_len] -> [1, pred_len, 1] -> [Batch, pred_len, 1]
        # 这样就能覆盖所有 Batch，并且在最后一维指明要取哪个分位数
        gather_indices = indices_tensor.view(1, pred_len, 1).expand(batch_size, -1, -1)
        
        # 执行 gather 操作 (dim=2 表示在分位数维度上取值)
        # 结果 shape: [Batch, pred_len, 1]
        selected_pred = torch.gather(full_pred, dim=2, index=gather_indices)
        
        # 5. 去除多余维度并转为 Numpy
        # shape: [Batch, pred_len]
        final_output = selected_pred.squeeze(-1)
        
        return final_output.detach().cpu().float().numpy()