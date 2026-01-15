import torch

class Model:
    def __init__(self, lookback_days=10000):
        self.lookback_days = lookback_days
        self.period = 24
        
    def forecast(self, horizon, inputs):
        """
        inputs: [Batch, Seq_Len]
        return: [Batch, horizon]
        """
        B, seq_len = inputs.shape
        
        # 1. 检查输入长度是否足够 (至少需要 1 天才能计算日平均，建议满足 lookback_days)
        days_available = seq_len // self.period
        days_to_use = min(self.lookback_days, days_available)
        
        if days_to_use < 1:
            raise ValueError(f"Input length ({seq_len}) is too short for Naive Daily Average (min 24)")

        # 2. 截取用于计算的历史数据 [Batch, days_to_use * 24]
        # 取数据的最后部分，确保时间是对齐的
        history_len = days_to_use * self.period
        history = inputs[:, -history_len:]
        
        # 3. 变形为 [Batch, Days, 24] 并计算均值 -> [Batch, 24]
        # 这里的 [Batch, 24] 代表了过去 N 天每一小时的平均值
        # 索引 0 对应 t-23 的平均，索引 23 对应 t 的平均
        # 预测 t+1 时，正好对应上一轮周期的索引 0，逻辑自洽
        history_reshaped = history.view(B, days_to_use, self.period)
        avg_daily_profile = torch.mean(history_reshaped, dim=1) 
        
        # 4. 拼接以匹配预测长度 horizon
        # 计算需要重复多少次才能覆盖 horizon
        repeats = (horizon + self.period - 1) // self.period
        
        # [Batch, 24 * repeats]
        predictions = avg_daily_profile.repeat(1, repeats)
        
        # 5. 截取所需的长度
        return predictions[:, :horizon]