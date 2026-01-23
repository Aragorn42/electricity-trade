import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from chinese_calendar import is_holiday

class PriceDataset(Dataset):
    def __init__(self, args, data, dates, stride=24, mode='test'):
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.stride = stride
        self.train_day_start = 730 - args.train_day - args.val_day - args.eval_day
        pd_dates = pd.to_datetime(dates) 

        # is_holiday() 对周末和法定节假日返回 True，对调休上班日返回 False
        # holiday_mask = [1 if is_holiday(d) else -1 for d in pd_dates]

        # 用1-7表示星期, 1 for monday, 2 for tuesday, ..., 7 for sunday
        holiday_mask = [d.weekday() + 1 for d in pd_dates]
        self.holiday_data = np.array(holiday_mask, dtype=np.float32)
        total_len = len(data)
        if mode == 'train':
            # 训练集：取前 train_day 天的数据
            start_idx = self.train_day_start * 24
            end_idx = start_idx + args.train_day * 24
            self.data_reset = data[start_idx:end_idx]
            self.holiday_reset = self.holiday_data[start_idx:end_idx]
            
        elif mode == 'val':
            # 验证集：介于 train 和 test 之间的数据
            start_idx = self.train_day_start * 24 + args.train_day * 24
            end_idx = total_len - args.eval_day * 24 - self.seq_len - self.pred_len + 24
            self.data_reset = data[start_idx:end_idx]
            self.holiday_reset = self.holiday_data[start_idx:end_idx]
            
        else:
            # 测试集：取最后 eval_day 天的数据
            start_idx = total_len - args.eval_day * 24 - self.seq_len - self.pred_len + 24
            end_idx = data.shape[0]
            self.data_reset = data[start_idx:]
            self.holiday_reset = self.holiday_data[start_idx:]
        print(f"{mode} day:{(end_idx-start_idx)/24}")
        
    def __len__(self):
        data_len = len(self.data_reset)
        total_window_size = self.seq_len + self.pred_len
        if data_len < total_window_size:
            return 0
        num_samples = (data_len - total_window_size) // self.stride + 1
        return num_samples
    
    def __getitem__(self, idx):
        actual_start_index = idx * self.stride
        x_end = actual_start_index + self.seq_len
        y_end = x_end + self.pred_len
        
        # 获取价格数据
        data_x = self.data_reset[actual_start_index : x_end]
        data_y = self.data_reset[x_end : y_end]
        
        # 获取假期特征 (0/1 数值)
        holiday_x = self.holiday_reset[actual_start_index : x_end]
        holiday_y = self.holiday_reset[x_end : y_end]
        
        # 返回全是 Tensor，可以直接输入模型
        # holiday_x 的形状是 (seq_len, )，如果需要拼接进模型，可能需要 unsqueeze
        return (torch.tensor(data_x, dtype=torch.float32),
                torch.tensor(data_y, dtype=torch.float32),
                torch.tensor(holiday_x, dtype=torch.float32), 
                torch.tensor(holiday_y, dtype=torch.float32))