from model import timesfm2_5time, timesfm2_5, naive_avg, holiday_avg, fixed
from utils.data_process import handle_excel, init_report_df, add_prediction_columns
import torch
from utils.dataset import PriceDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np
import pandas as pd
from os import path as file

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

def forecast(df, args):
    '''
    file: pd.DataFrame with 'Datetime' and 'Price' columns
    '''
    da_values = df['Price'].values.astype(np.float32)
    da_dates = df['Datetime'].values
    dataset = PriceDataset(da_values, da_dates, args.seq_len, args.pred_len, stride=24, mode='test')
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    
    model = get_model(args)
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
            else:
                y_pred = model.forecast(args.pred_len, inputs_for_model)

            y_pred = y_pred[0] if isinstance(y_pred, tuple) else y_pred
            
            # 如果进行了填充，需要把填充的多余部分切掉
            if current_batch_size < args.batchsize:
                y_pred = y_pred[:current_batch_size]
                
            y_preds.append(y_pred)
            
        # 合并所有 Batch
        y_preds = np.concatenate(y_preds, axis=0) # [Total_Samples, pred_len]
        y_preds = y_preds[:, -24:]
        y_preds_ordered = y_preds.flatten()
        return y_preds_ordered
    
def evaluate(args):
    # df with datatime and Price
    da_raw, rt_raw, diff_raw = handle_excel(args.file_path)
    if file.exists("output_report.xlsx"):
        df_report = pd.read_excel("output_report.xlsx", header=None)
    else:
        df_report = init_report_df(args, da_raw, rt_raw, diff_raw)

    da_preds = forecast(da_raw, args)
    rt_preds = forecast(rt_raw, args)
    diff_preds = forecast(diff_raw, args)
    diff_true = diff_raw.iloc[1:, 1].values
    # df_report, preds, trues, args
    # this function compares the sign of preds and trues and add 2 column after df_report
    df_report = add_prediction_columns(df_report, diff_preds, diff_true, args, is_two_variate=False)
    if args.two_variate:
        df_report = add_prediction_columns(df_report, da_preds - rt_preds, diff_true, args, is_two_variate=True)

    # # pandas >= 2.1
    # # the actual column names is not same as the column in excel

    df_report = df_report.map(
        lambda x: x if not isinstance(x, (float, int)) else "{:.3f}".format(x)
    )
    for col in df_report.columns:
        if "准确率" in str(df_report[col].iloc[0]):
            df_report[col].iloc[1:] = df_report[col].iloc[1:].map(\
                lambda x: "{:.0f}".format(float(x))
            )
    
    df_report.to_excel("output_report.xlsx", index=False, header=False)
