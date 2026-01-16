from model import timesfm2_5time, timesfm2_5, naive_avg, holiday_avg, fixed, DLinear, PatchTST
from utils.data_process import handle_excel, init_report_df, add_prediction_columns
import torch
from utils.dataset import PriceDataset
from torch.utils.data import DataLoader
from exp.train import train
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from os import path as file

def get_model(args):
    if args.model_type == "TimesFM-2.5time":
        model = timesfm2_5time.Model(model_path=args.model_path)
    elif args.model_type == "TimesFM-2.5":
        model = timesfm2_5.Model(model_path=args.model_path)
    elif args.model_type == "NaiveAvg":
        model = naive_avg.Model()
    elif args.model_type == "HolidayAvg":
        model = holiday_avg.Model()
    elif args.model_type == "fixed":
        value1 = torch.tensor([
            -42.352, -16.900, -5.879, 8.722, 19.356,16.879, -13.520, -4.291, -22.837, -6.047, 19.049, 6.494, 30.489, -9.075, 21.705, 11.469, 2.213, -1.240, -6.075, -15.271, -23.021, -26.439, -22.399, 4.697
        ])
        # 2024.1.1 to 2025.5.31, diff average value
        value2 = torch.tensor([
            0.339, 0.181, 0.250, 0.066, 0.139, 0.329, 0.211, -0.172, 0.000, 0.000, -0.003, -0.581, -1.096, -0.426, -0.024, -0.979, -0.787, -1.188, -0.324, 0.368, 0.349, -0.191, -0.152, -0.190
        ])
        model = fixed.Model(value1)
    elif args.model_type == "DLinear":
        model = DLinear.Model(args)
    elif args.model_type == "PatchTST":
        model = PatchTST.Model(args)
    return model

def scaled_data(df):
    scaler = StandardScaler()
    df_scaled = df.copy()
    # 提取价格列(假设为第2列, index=1)并reshape为(N, 1)
    prices = df.iloc[:, 1].values.reshape(-1, 1)
    # 归一化并放回dataframe
    df_scaled.iloc[:, 1] = scaler.fit_transform(prices).flatten()
    return df_scaled, scaler
    
def forecast(model, df, args):
    '''
    file: pd.DataFrame with 'Datetime' and 'Price' columns
    '''
    da_values = df['Price'].values.astype(np.float32)
    da_dates = df['Datetime'].values
    dataset = PriceDataset(args, da_values, da_dates, stride=24, mode='test')
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    
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

            inputs_for_model = x_padded

            if args.model_type == "HolidayAvg":
                y_pred = model.forecast(
                    horizon=args.pred_len, 
                    history_x=inputs_for_model,   # [batch, Seq]
                    holiday_x=x_holiday_padded,   # [batch, Seq]
                    holiday_y=y_holiday_padded    # [batch, Pred]
                )
            elif args.model_type == "DLinear" or args.model_type == "PatchTST":
                y_pred = model(inputs_for_model.to(torch.device('cuda')).unsqueeze(2))
            else:
                y_pred = model.forecast(args.pred_len, inputs_for_model)

            y_pred = y_pred[0] if isinstance(y_pred, tuple) else y_pred
            
            # 如果进行了填充，需要把填充的多余部分切掉
            if current_batch_size < args.batchsize:
                y_pred = y_pred[:current_batch_size]
            y_preds.append(y_pred.detach().cpu().numpy())
            
        # 合并所有 Batch
        y_preds = np.concatenate(y_preds, axis=0) # [Total_Samples, pred_len]
        y_preds = y_preds[:, -24:]
        y_preds_ordered = y_preds.flatten()
        return y_preds_ordered
    
def evaluate(args):
    # df with datatime and Price
    da_raw, rt_raw, diff_raw = handle_excel(args.file_path)
    da, da_scaler = scaled_data(da_raw)
    rt, rt_scaler = scaled_data(rt_raw)
    diff, diff_scaler = scaled_data(diff_raw)
    if file.exists("output_report.xlsx"):
        df_report = pd.read_excel("output_report.xlsx", header=None)
    else:
        df_report = init_report_df(args, da_raw, rt_raw, diff_raw)
    device = torch.device('cuda')
    if args.need_train:
        da_preds = forecast(train(args, get_model(args).to(device), da), da, args)
        rt_preds = forecast(train(args, get_model(args).to(device), rt), rt, args)
        diff_preds = forecast(train(args, get_model(args).to(device), diff), diff, args)
    else:
        model = get_model(args)
        da_preds = forecast(model, da, args)
        rt_preds = forecast(model, rt, args)
        diff_preds = forecast(model, diff, args)
    da_preds = da_scaler.inverse_transform(da_preds.reshape(-1, 1)).flatten()
    rt_preds = rt_scaler.inverse_transform(rt_preds.reshape(-1, 1)).flatten()
    diff_preds = diff_scaler.inverse_transform(diff_preds.reshape(-1, 1)).flatten()

    diff_true = diff_raw.iloc[1:, 1].values
    # df_report, preds, trues, args
    # this function compares the sign of preds and trues and add 2 column after df_report
    df_report = add_prediction_columns(df_report, diff_preds, diff_true, args, is_two_variate=False)
    if args.two_variate:
        df_report = add_prediction_columns(df_report, da_preds - rt_preds, diff_true, args, is_two_variate=True)

    output_file = "output_report.xlsx"
    # 使用 xlsxwriter 引擎
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        df_report.to_excel(writer, index=False, header=False, sheet_name='Sheet1')
        workbook  = writer.book
        worksheet = writer.sheets['Sheet1']
        format_float_3 = workbook.add_format({'num_format': '0.000'})
        format_int = workbook.add_format({'num_format': '0'})
        for col_idx, col_name in enumerate(df_report.columns):
            first_cell_val = str(df_report.iloc[0, col_idx])
            if "准确率" in first_cell_val:
                worksheet.set_column(col_idx, col_idx, 15, format_int)
            else:
                worksheet.set_column(col_idx, col_idx, 15, format_float_3)
