import timesfm
import torch
import math
import numpy as np

class Model():
    def __init__(self, model_path):
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_path, 
            local_files_only=True
        )
        self.model.compile(timesfm.ForecastConfig(
            max_context=64,
            max_horizon=32, 
            normalize_inputs=True,
            use_continuous_quantile_head=True, # 概率预测
            force_flip_invariance=True, # 输入-X保证输出-Y
            infer_is_positive=False, # 强制输出非负
            fix_quantile_crossing=False # 分位数保序
        ))

    def forecast(self, pred_len, inputs, args, quants=None):
        """
        基于 self.model.forecast(pred_len, inputs) 接口实现
        inputs: [Batch, seq_Len]
        """
        B, seq_len = inputs.shape
        period = 24
        num_days = seq_len // period
        pred_steps = math.ceil(pred_len / period)
        
        quants =  [1, 6, 6, 8, 6, 6, 7, 6, 7, 7, 6, 6, 7, 6, 0, 6, 6, 7, 6, 6, 7, 9, 6, 9]
        x_reshaped = inputs[:, :num_days * period].contiguous().view(B, num_days, period)
        
        hourly_results = []
        # 2. 重塑输入数据: [Batch, Days, 24]
        # 3. 按小时循环
        for h in range(period):
            # 取出当前小时的 Tensor [Batch, num_days]
            input_h_tensor = x_reshaped[:, :, h]
            
            # --- 【关键修复】 ---
            # TimesFM 的 forecast 接口通常要求输入是 List of Numpy Arrays
            # 1. 转为 Numpy
            input_h_numpy = input_h_tensor.detach().cpu().numpy()
            # 2. 转为 List (也就是 list(array) 会变成 [row1, row2, ...])
            input_h_list = list(input_h_numpy)
            # ------------------
            
            # 调用模型 (传入 List)
            # 输出通常是 (mean, full_quantiles_numpy)
            _, y_pred_h = self.model.forecast(pred_steps, input_h_list)
            
            # --- 结果处理 ---
            # TimesFM 返回的 y_pred_h 通常是 numpy array
            # 如果是 numpy，转回 Tensor 以便后续 stack
            if isinstance(y_pred_h, np.ndarray):
                y_pred_h = torch.from_numpy(y_pred_h).to(inputs.device)
            
            # y_pred_h shape: [Batch, pred_steps, quantiles]
            current_quant_idx = quants[h]
            output_h = y_pred_h[:, :, current_quant_idx] # [Batch, pred_steps]
            
            hourly_results.append(output_h)
            
        # 4. 拼接
        forecast_stacked = torch.stack(hourly_results, dim=1) # [Batch, 24, pred_steps]
        forecast_ordered = forecast_stacked.permute(0, 2, 1)  # [Batch, pred_steps, 24]
        
        # 5. 展平并截断
        final_output = forecast_ordered.contiguous().view(B, -1)
        final_output = final_output[:, :pred_len]
        
        # 根据 Eval 代码的要求，通常需要返回 Numpy
        return final_output.detach().cpu().numpy()