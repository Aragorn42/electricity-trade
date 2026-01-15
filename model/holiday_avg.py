import torch

class Model:
    def __init__(self, lookback_days=10000):
        self.lookback_days = lookback_days
        self.period = 24  # 每日24小时

    def forecast(self, horizon, history_x, holiday_x, holiday_y):
        """
        params:
            horizon: 预测步长 (int)
            history_x: [Batch, Seq_Len] (数值)
            holiday_x: [Batch, Seq_Len] (0=工作日, 1=假期)
            holiday_y: [Batch, Horizon] (未来的假期情况)
        """
        B, seq_len = history_x.shape
        
        # 1. 确定可用的天数
        days_available = seq_len // self.period
        days_to_use = min(self.lookback_days, days_available)
        
        if days_to_use < 1:
            raise ValueError("History too short to compute daily average.")

        # 截取需要用到的最近历史数据
        # [Batch, days_to_use * 24]
        history_len = days_to_use * self.period
        hist_vals = history_x[:, -history_len:]
        hist_flags = holiday_x[:, -history_len:]
        
        # 2. 变形为 [Batch, Days, 24] 以便按天聚合
        # 这里的 dim=2 (长度24) 就对应了 0点...23点
        vals_daily = hist_vals.view(B, days_to_use, self.period)
        flags_daily = hist_flags.view(B, days_to_use, self.period)
        
        # [Batch, Days, 1]
        day_is_holiday = (flags_daily.mean(dim=2, keepdim=True) > 0.5).float()
        day_is_workday = 1.0 - day_is_holiday
        
        # 3. 计算两套 Profile (核心逻辑)
        # 形状都是 [Batch, 24]，分别代表 0-23 点的均值
        
        # --- 计算工作日 Profile ---
        sum_vals_work = (vals_daily * day_is_workday).sum(dim=1) 
        count_work = day_is_workday.sum(dim=1) # [Batch, 1]
        
        # --- 计算假期 Profile ---
        sum_vals_holi = (vals_daily * day_is_holiday).sum(dim=1)
        count_holi = day_is_holiday.sum(dim=1)
        
        # --- 兜底逻辑 (Fallback) ---
        # 计算全局平均 (不管假期还是工作日)，用于填补缺失情况
        # 比如：如果历史全是工作日，那 count_holi 就是 0，不能除
        global_avg_profile = vals_daily.mean(dim=1) # [Batch, 24]
        
        # 如果 count > 0 用计算出的均值，否则用全局均值代替
        # 加上 1e-6 防止除以0报错
        work_profile = torch.where(
            count_work > 0, 
            sum_vals_work / (count_work + 1e-6), 
            global_avg_profile
        )
        
        holi_profile = torch.where(
            count_holi > 0, 
            sum_vals_holi / (count_holi + 1e-6), 
            global_avg_profile
        )
        
        # 4. 根据未来的 holiday_y 组装预测结果
        # holiday_y: [Batch, Horizon]

        repeats = (horizon + self.period - 1) // self.period
        
        # [Batch, 24 * repeats]
        full_work_pred = work_profile.repeat(1, repeats)
        full_holi_pred = holi_profile.repeat(1, repeats)
        
        # 截取到 horizon 长度 [Batch, Horizon]
        full_work_pred = full_work_pred[:, :horizon]
        full_holi_pred = full_holi_pred[:, :horizon]
        
        # 5. 最终选择
        # holiday_y 为 1 处取 holi_pred，为 0 处取 work_pred
        # 这里的 holiday_y 对应了未来具体的每一个小时
        final_pred = full_holi_pred * holiday_y + full_work_pred * (1 - holiday_y)
        
        return final_pred