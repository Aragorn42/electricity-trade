import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from chinese_calendar import is_holiday

class PriceDataset(Dataset):
    def __init__(self, args, data, dates, input_len, pred_len, stride=24, mode='test'):
        self.input_len = input_len
        self.pred_len = pred_len
        self.stride = stride
        
        pd_dates = pd.to_datetime(dates) 

        # is_holiday() 对周末和法定节假日返回 True，对调休上班日返回 False
        holiday_mask = [1.0 if is_holiday(d) else 0.0 for d in pd_dates]

        # df_check = pd.DataFrame({
        #     "Date": pd_dates,
        #     "Is_Holiday": holiday_mask
        # })
        # df_check.to_csv("holiday_check.csv", index=False)

        self.holiday_data = np.array(holiday_mask, dtype=np.float32)
        
        total_len = len(data)
        test_day = args.eval_day
        
        start_idx = total_len - test_day * 96 - input_len - pred_len + 96
        
        # 数据切片
        self.data_reset = data[start_idx:]
        # 日期特征切片 (也进行相同的切片)
        self.holiday_reset = self.holiday_data[start_idx:]
        
    def __len__(self):
        data_len = len(self.data_reset)
        total_window_size = self.input_len + self.pred_len
        if data_len < total_window_size:
            return 0
        num_samples = (data_len - total_window_size) // self.stride + 1
        return num_samples
    
    def __getitem__(self, idx):
        actual_start_index = idx * self.stride
        x_end = actual_start_index + self.input_len
        y_end = x_end + self.pred_len
        
        # 获取价格数据
        data_x = self.data_reset[actual_start_index : x_end]
        data_y = self.data_reset[x_end : y_end]
        
        # 获取假期特征 (0/1 数值)
        holiday_x = self.holiday_reset[actual_start_index : x_end]
        holiday_y = self.holiday_reset[x_end : y_end]
        
        # 返回全是 Tensor，可以直接输入模型
        # holiday_x 的形状是 (input_len, )，如果需要拼接进模型，可能需要 unsqueeze
        return (torch.tensor(data_x, dtype=torch.float32),
                torch.tensor(data_y, dtype=torch.float32),
                torch.tensor(holiday_x, dtype=torch.float32), 
                torch.tensor(holiday_y, dtype=torch.float32))