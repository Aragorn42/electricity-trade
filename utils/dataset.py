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
        pd_dates = pd.to_datetime(dates) 

        # 日期参数格式：YYYY-MM-DD，按天定位并映射到小时级索引切片
        train_start_day = pd.to_datetime(args.train_start_day).date()
        train_end_day = pd.to_datetime(args.train_end_day).date()
        val_start_day = pd.to_datetime(args.val_start_day).date()
        val_end_day = pd.to_datetime(args.val_end_day).date()
        eval_start_day = pd.to_datetime(args.eval_start_day).date()
        eval_end_day = pd.to_datetime(args.eval_end_day).date()

        date_only = pd_dates.date

        def get_range_idx(start_day, end_day, split_name):
            idx = np.where((date_only >= start_day) & (date_only <= end_day))[0]
            if len(idx) == 0:
                raise ValueError(
                    f"{split_name} date range [{start_day}, {end_day}] has no overlap with provided dates"
                )
            return idx[0], idx[-1] + 1

        def get_context_range_idx(start_day, end_day, split_name):
            split_start, split_end = get_range_idx(start_day, end_day, split_name)
            # 前置上下文增加 seq_len + pred_len - 24，使首个样本预测窗口最后24点对齐 split 起始日
            pre_context = self.seq_len + self.pred_len - 24
            context_start = max(0, split_start - pre_context)
            return context_start, split_end

        # is_holiday() 对周末和法定节假日返回 True，对调休上班日返回 False
        holiday_mask = [1 if is_holiday(d) else 0 for d in pd_dates]

        self.holiday_data = np.array(holiday_mask, dtype=np.float32)
        if mode == 'train':
            # 训练集：按日期切片，并在前面补 seq_len + pred_len - 24 长度上下文
            start_idx, end_idx = get_context_range_idx(train_start_day, train_end_day, 'train')
            self.data_reset = data[start_idx:end_idx]
            self.holiday_reset = self.holiday_data[start_idx:end_idx]
            
        elif mode == 'val':
            # 验证集：按日期切片，并在前面补 seq_len + pred_len - 24 长度上下文
            start_idx, end_idx = get_context_range_idx(val_start_day, val_end_day, 'val')
            self.data_reset = data[start_idx:end_idx]
            self.holiday_reset = self.holiday_data[start_idx:end_idx]
            
        else:
            # 测试集：按日期切片，并在前面补 seq_len + pred_len - 24 长度上下文
            start_idx, end_idx = get_context_range_idx(eval_start_day, eval_end_day, 'eval')
            self.data_reset = data[start_idx:end_idx]
            self.holiday_reset = self.holiday_data[start_idx:end_idx]
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