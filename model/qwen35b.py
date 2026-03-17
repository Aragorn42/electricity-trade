import json

import torch
import math
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from llamafactory.chat import ChatModel

class Model:
    def __init__(self, args):
        # 建议在 args 中传入基础模型和微调后 LoRA 权重的路径
        base_model_path =  "/data/liym/ms_cache/models/Qwen/Qwen3___5-35B-A3B"
        adapter_path = "/home/liym/code/fine_tune/saves/Qwen3.5-35B-A3B-Thinking/lora/train_2026-03-10-23-38-01/checkpoint-40"
        self.period = 24
        args = dict(
            model_name_or_path=base_model_path,
            #adapter_name_or_path=adapter_path,
            template='qwen',
            infer_backend="vllm",
            infer_dtype="bfloat16",
            temperature=0.01,
            top_p=1.0,
            do_sample=False, 
        )
        # vllm limit_mm_per_prompt={"image": 3, "video": 1}
        self.model = ChatModel(args)
        print("Model Loaded Successfully")

    def forecast(self, pred_len, inputs, args):
        """
        inputs: [Batch, seq_Len] (Tensor)
        outputs: [Batch, pred_len] (Tensor)
        """
        B, seq_len = inputs.shape
        
        # --- 1. 传统的数据截断与 Reshape 逻辑 (保持不变) ---
        remainder = seq_len % self.period
        if remainder != 0:
            inputs = inputs[:, remainder:]
            seq_len = inputs.shape[1]
            
        num_days = seq_len // self.period
        pred_steps = math.ceil(pred_len / self.period)
        
        # [Batch, num_days, 24]
        x_reshaped = inputs.contiguous().view(B, num_days, self.period)
        # -> [Batch, 24, num_days] -> [Batch * 24, num_days]
        inputs_daily = x_reshaped.permute(0, 2, 1).reshape(B * self.period, num_days)


        generated_texts = []
        for i in range(B * self.period):
            # 动态打印进度条
            print(f"Predicting sample {i+1}/{B * self.period}", end="\r")
            
            # 获取当前是哪一天、哪一小时的数据
            hour_idx = i % self.period
            history_values = inputs_daily[i].tolist()
            # 格式化历史数值，保留两位小数并用逗号分隔
            history_str = ", ".join([f"{val:.2f}" for val in history_values])
            
            system_prompt = (
                f"你是一个时间序列预测助手。根据输入的历史数值，预测未来 {pred_steps} 天同一时刻的数值。这段数据是电力交易市场当中日前价格和实时价格的差值。\n"
                f"输出要求：只输出 {pred_steps} 个预测数值，用逗号分隔，不要有任何解释文字、不要有换行、不要有其他字符。"
            )
            
            # 2. 构建 User Prompt (messages 列表只放对话内容)
            user_prompt = (
                f"时间点：{hour_idx} 点\n"
                f"历史数据：第 1 天 至 第 {num_days} 天 ({num_days} 天)\n"
                f"历史数值（{num_days} 个点，按时间顺序）：{history_str}\n"
                f"请预测后 {pred_steps} 天 {hour_idx} 点的数值（{pred_steps} 个数字），只输出数字，用逗号分隔："
            )
        
            # messages 列表里不要包含 {"role": "system"}
            messages = [
                {"role": "user", "content": user_prompt}
            ]
            
            # 3. 在调用 chat 时，通过 system= 参数传入
            response = self.model.chat(messages, system=system_prompt)
            
            # response[0].response_text 直接就是预测出来的字符串（如 "12.34, 15.67"）
            generated_texts.append(response[0].response_text)
            
        print("\n预测完成！")
        
        parsed_results = []
        for idx, text in enumerate(generated_texts):
            nums = re.findall(r"-?\d+\.?\d*", text)
            nums_float = [float(n) for n in nums][:pred_steps]
            
            if len(nums_float) < pred_steps:
                print('output less than expected, applying padding with last known value.')
                last_known_val = nums_float[-1] if nums_float else inputs_daily[idx, -1].item()
                nums_float.extend([last_known_val] * (pred_steps - len(nums_float)))
                
            parsed_results.append(nums_float)

        outputs_tensor = torch.tensor(parsed_results, device=inputs.device, dtype=inputs.dtype)
        forecast_reshaped = outputs_tensor.view(B, self.period, pred_steps)
        forecast_ordered = forecast_reshaped.permute(0, 1, 2).transpose(1, 2)
        
        # 展平 [Batch, Steps * 24]
        final_output = forecast_ordered.contiguous().reshape(B, pred_steps * self.period)
        
        return final_output