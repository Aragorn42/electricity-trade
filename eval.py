from model import timesfm2_5time, timesfm2_5, naive_avg, holiday_avg, fixed
from utils.data_process import handle_excel, init_report_df, add_prediction_columns
import torch
from utils.dataset import PriceDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np
import pandas as pd
from os import path as file
import jax
jax.config.update("jax_log_compiles", True)
def get_model(args):
    if args.model_type == "200MTime":
        model = timesfm2_5time.Model(model_path=args.model_path)
    elif args.model_type == "200M":
        model = timesfm2_5.Model(model_path=args.model_path)
    elif args.model_type == "NaiveAvg":
        model = naive_avg.Model()
    elif args.model_type == "HolidayAvg":
        model = holiday_avg.Model()
    elif args.model_type == "fixed":
        # 2024.1.1 to 2025.5.31, diff average value
        value = torch.tensor([
            0.339, 0.181, 0.250, 0.066, 0.139, 0.329, 0.211, -0.172, 0.000, 0.000, -0.003, -0.581, -1.096, -0.426, -0.024, -0.979, -0.787, -1.188, -0.324, 0.368, 0.349, -0.191, -0.152, -0.190
        ])
        model = fixed.Model(value)
    return model

def forecast(model, df, args):
    '''
    file: pd.DataFrame with 'date' and 'OT' columns
    '''
    da_values = df['OT'].values.astype(np.float32)
    da_dates = df['date'].values
    dataset = PriceDataset(args, da_values, da_dates, args.seq_len, args.pred_len, stride=1, mode='test')
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    
    #model = get_model(args)
    y_preds = []
    with torch.no_grad():
        for x, y_true, x_holiday, y_holiday in dataloader:            
            current_batch_size = x.shape[0]

            # 如果是最后一个 Batch 且不足 BATCH_SIZE，填充 x
            if current_batch_size < args.batchsize:
                pad_len = args.batchsize - current_batch_size

                padding_x = torch.zeros(pad_len, x.shape[1], device=x.device, dtype=x.dtype)
                x_padded = torch.cat([x, padding_x], dim=0)
                
                padding_xh = torch.zeros(pad_len, x_holiday.shape[1], device=x_holiday.device, dtype=x_holiday.dtype)
                x_holiday_padded = torch.cat([x_holiday, padding_xh], dim=0)
                
                padding_yh = torch.zeros(pad_len, y_holiday.shape[1], device=y_holiday.device, dtype=y_holiday.dtype)
                y_holiday_padded = torch.cat([y_holiday, padding_yh], dim=0)
                
            else:
                x_padded = x
                x_holiday_padded = x_holiday
                y_holiday_padded = y_holiday

            inputs_for_model = x_padded.cpu().detach().numpy()

            if args.model_type == "HolidayAvg":
                y_pred = model.forecast(
                    horizon=args.pred_len, 
                    history_x=inputs_for_model,   # [batch, Seq]
                    holiday_x=x_holiday_padded,   # [batch, Seq]
                    holiday_y=y_holiday_padded    # [batch, Pred]
                )
            else:
                y_pred = model.forecast(args.pred_len, inputs_for_model)

            y_pred = y_pred[0] if isinstance(y_pred, tuple) else y_pred
            
            # 如果进行了填充，需要把填充的多余部分切掉
            if current_batch_size < args.batchsize:
                y_pred = y_pred[:current_batch_size]
                
            y_preds.append(y_pred)
            
        # 合并所有 Batch
        y_preds = np.concatenate(y_preds, axis=0) # [Total_Samples, pred_len]
        y_preds = y_preds[:, -1:]
        y_preds_ordered = y_preds.flatten()
        return y_preds_ordered
    
def evaluate(args):
    # df with datatime and Price
    df = pd.read_csv("dataset/skippd.csv")
    model = get_model(args)
    df_pred = forecast(model, df, args)
    print(df_pred.shape)
    df_true = df.iloc[-df_pred.shape[0]:, 1].values
    return df_pred
import pandas as pd
import numpy as np
import argparse

# 假设你的 evaluate 函数定义如下（保持不变）
# def evaluate(args): ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='200M', help='Model type: 200M, HolidayAvg, NaiveAvg, 200MTime')

    parser.add_argument('--model_path', type=str, default="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/TimesFM2_5_200M")
    parser.add_argument('--file_path', type=str, default='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm', help='Path to the Excel file')
    parser.add_argument('--seq_len', type=int, default=336, help='Input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='Prediction length')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--two_variate', type=bool, default=True, help='Whether to perform two-variate evaluation')
    parser.add_argument('--eval_day', type=int, default=300, help='Number of days to evaluate')
    # 确保加上 batchsize, seq_len 等必须参数的默认值，否则 evaluate 可能会报错

    
    args = parser.parse_args()

    original_file_path = "dataset/skippd.csv"
    df_final = pd.read_csv(original_file_path)
    total_rows = len(df_final)

    # 用于记录最长的预测长度，以便最后裁剪
    max_pred_len = 0
    pred_col_names = []

    # 2. 循环预测
    pred_lengths = [1, 16, 96]
    
    for x in pred_lengths:
        print(f"正在进行 pred_len={x} 的推理...")
        args.pred_len = x
        
        # 获取结果 (10369,)
        y_pred = evaluate(args) 
        y_pred = np.array(y_pred).flatten()
        
        current_pred_len = len(y_pred)
        
        # 更新最大长度记录
        if current_pred_len > max_pred_len:
            max_pred_len = current_pred_len
            
        col_name = f'pred_{x}'
        pred_col_names.append(col_name)
        
        # 初始化列
        df_final[col_name] = np.nan
        
        # 从末尾对齐填充
        # 如果预测长度比原始数据还长，取原始数据长度；否则取预测长度
        safe_len = min(total_rows, current_pred_len)
        df_final.iloc[-safe_len:, df_final.columns.get_loc(col_name)] = y_pred[-safe_len:]

    # 3. 【核心修改】删除前面没有填入的行
    # 直接保留最后 max_pred_len 行
    if max_pred_len > 0:
        print(f"正在裁剪数据，只保留最后 {max_pred_len} 行...")
        df_final = df_final.iloc[-max_pred_len:].copy() # copy避免警告
    else:
        print("警告：没有产生任何预测数据。")

    # 4. 保存为 Excel
    output_path = "eval_results_merged11696.xlsx"
    print(f"正在保存 {df_final.shape[0]} 行数据到 {output_path} ...")
    df_final.to_excel(output_path, index=False)
    print("完成。")