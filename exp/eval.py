#from model import #timesfm2_5time, timesfm2_5, naive_avg, holiday_avg, fixed, DLinear, PatchTST, TimeMoE#, chronos2, chronos2time, chronos2holiday, YingLong, chronos2time2
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
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def get_model(args):
    if "TimesFM-2.5time" in args.model_type:
        from model import timesfm2_5time
        model = timesfm2_5time.Model(model_path=args.model_path)
    elif "TimesFM-2.5" in args.model_type:
        from model import timesfm2_5
        model = timesfm2_5.Model(model_path=args.model_path)
    elif args.model_type == "NaiveAvg":
        from model import naive_avg
        model = naive_avg.Model()
    elif args.model_type == "HolidayAvg":
        from model import holiday_avg
        model = holiday_avg.Model()
    elif args.model_type == "fixed":
        from model import fixed
        # value1 = [1]*24
        # value2 = [1]*24
        # # 0, 1, 8, 18, 19, 20, 21, 22等于-1
        # value1[0], value1[1], value1[8], value1[18], value1[19], value1[20], value1[21], value1[22] = -1, -1, -1, -1, -1, -1, -1, -1
        # # 0, 1, 2, 3, 13, 16, 17, 18, 19, 20, 21, 22, 23等于-1
        # value2[0], value2[1], value2[2], value2[3], value2[13], value2[16], value2[17], value2[18], value2[19], value2[20], value2[21], value2[22], value2[23] = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        value1, value2 = [1]*24, [1]*24
        # 0, 8为-1
        value1[0], value1[8] = -1, -1
        # 0, 1, 13, 18, 19, 20, 21, 22
        value2[0], value2[1], value2[13], value2[18], value2[19], value2[20], value2[21], value2[22] = -1, -1, -1, -1, -1, -1, -1, -1
        model = fixed.Model(value1, value2)
    elif args.model_type == "DLinear":
        from model import DLinear
        model = DLinear.Model(args)
    elif args.model_type == "PatchTST":
        from model import PatchTST
        model = PatchTST.Model(args)
    elif "Chronos-2time2" in args.model_type:
        from model import chronos2time2
        model = chronos2time2.Model(args)
    elif "Chronos-2holiday" in args.model_type:
        from model import chronos2holiday
        model = chronos2holiday.Model(args)
    elif "Chronos-2time" in args.model_type:
        from model import chronos2time
        model = chronos2time.Model(args)
    elif "Chronos-2" in args.model_type:
        from model import chronos2
        model = chronos2.Model(args)
    elif args.model_type == "YingLong":
        from model import YingLong
        model = YingLong.Model(args)
    elif args.model_type == "YingLongtime":
        from model import YingLongtime
        model = YingLongtime.Model(args)
    elif args.model_type == "TimeMoE":
        from model import TimeMoE
        model = TimeMoE.Model(args)
    elif args.model_type == "Timer":
        from model import Timer
        model = Timer.Model()
    elif args.model_type == "FalconTST":
        from model import FalconTST
        model = FalconTST.Model()
    elif args.model_type == "moirai2time2":
        from model import Moirai2time2
        model = Moirai2time2.Model()
    elif "moirai" in args.model_type.lower():
        from model import Moirai
        model = Moirai.Model(args.model_type)
    elif "sundial" in args.model_type.lower():
        from model import sundial
        model = sundial.Model()
    elif "holiday_avg_workday" in args.model_type.lower():
        from model import holiday_avg_workday
        model = holiday_avg_workday.Model()
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

            if args.model_type == "HolidayAvg" or args.model_type == "Chronos-2holiday" or args.model_type == "fixed":
                y_pred = model.forecast(
                    args, 
                    inputs_for_model,   # [batch, Seq]
                    holiday_x=x_holiday_padded,   # [batch, Seq]
                    holiday_y=y_holiday_padded    # [batch, Pred]
                )
            elif args.model_type == "DLinear" or args.model_type == "PatchTST":
                y_pred = model(inputs_for_model.to(torch.device('cuda')).unsqueeze(2))
                y_pred = y_pred.detach().cpu().numpy()
            else:
                y_pred = model.forecast(args.pred_len, inputs_for_model, args)
            #print(quant.shape)
            #y_pred = y_pred[0] if isinstance(y_pred, tuple) else y_pred
            # 如果进行了填充，需要把填充的多余部分切掉
            if current_batch_size < args.batchsize:
                y_pred = y_pred[:current_batch_size]
            y_preds.append(y_pred)
            
        # 合并所有 Batch
        y_preds = np.concatenate(y_preds, axis=0) # [Total_Samples, pred_len]
        y_preds = y_preds[:, -24:]
        y_preds_ordered = y_preds.flatten()
        return y_preds_ordered
    
import numpy as np
import pandas as pd


def weekday_sign_accuracy(preds, trues, dates=None, weekday_mask=None):
    """
    工作日符号准确率：只在工作日样本上比较 sign(preds) == sign(trues)

    Args:
        preds: array-like
        trues: array-like
        dates: array-like 日期序列（可选）
        weekday_mask: array-like bool 掩码（可选），True 表示保留该样本

    Returns:
        float: 准确率；若无有效样本返回 np.nan
    """
    preds = np.asarray(preds).reshape(-1)
    trues = np.asarray(trues).reshape(-1)

    if preds.shape[0] != trues.shape[0]:
        raise ValueError("preds 和 trues 长度必须一致")

    if weekday_mask is not None:
        mask = np.asarray(weekday_mask).reshape(-1)
        if mask.dtype != bool:
            mask = mask.astype(bool)
        if mask.shape[0] != preds.shape[0]:
            raise ValueError("weekday_mask 长度必须与 preds/trues 一致")
    elif dates is not None:
        arr = np.asarray(dates).reshape(-1)
        if arr.shape[0] != preds.shape[0]:
            raise ValueError("dates 长度必须与 preds/trues 一致")

        # 如果用户误传了布尔数组，直接按掩码处理，不再 to_datetime
        if arr.dtype == bool:
            mask = arr
        else:
            dt = pd.to_datetime(arr, errors="coerce")
            mask = (dt.weekday < 5) & (~pd.isna(dt))
    else:
        raise ValueError("dates 和 weekday_mask 至少提供一个")

    if mask.sum() == 0:
        return np.nan

    pred_sign = np.sign(preds[mask])
    true_sign = np.sign(trues[mask])
    return float((pred_sign == true_sign).mean())

def evaluate(args):
    # df with datatime and Price
    da_raw, rt_raw, diff_raw = handle_excel(args.file_path)
    da, da_scaler = scaled_data(da_raw)
    rt, rt_scaler = scaled_data(rt_raw)
    diff, diff_scaler = scaled_data(diff_raw)
    # if file.exists("output_report.xlsx"):
    #     df_report = pd.read_excel("output_report.xlsx", header=None)
    # else:
    #     df_report = init_report_df(args, da_raw, rt_raw, diff_raw)
    device = torch.device('cuda')
    if args.need_train:
        #da_preds = forecast(train(args, get_model(args).to(device), da), da, args)
        #rt_preds = forecast(train(args, get_model(args).to(device), rt), rt, args)
        diff_preds = forecast(train(args, get_model(args).to(device), diff), diff, args)
    else:
        model = get_model(args)
        # da_preds = forecast(model, da, args)
        # rt_preds = forecast(model, rt, args)
        diff_preds = forecast(model, diff, args)
    # da_preds = da_scaler.inverse_transform(da_preds.reshape(-1, 1)).flatten()
    # rt_preds = rt_scaler.inverse_transform(rt_preds.reshape(-1, 1)).flatten()
    diff_preds = diff_scaler.inverse_transform(diff_preds.reshape(-1, 1)).flatten()

    diff_true = diff_raw.iloc[1:, 1].values
    # df_report, preds, trues, args
    # this function compares the sign of preds and trues and add 2 column after df_report
    # df_report = add_prediction_columns(df_report, diff_preds, diff_true, args, is_two_variate=False)
    # if args.two_variate:
    #     df_report = add_prediction_columns(df_report, da_preds - rt_preds, diff_true, args, is_two_variate=True)
    eval_start_day = pd.to_datetime(args.eval_start_day).date()
    eval_end_day = pd.to_datetime(args.eval_end_day).date()

    datetime_col = 'Datetime' if 'Datetime' in diff_raw.columns else 'Datatime'
    if datetime_col not in diff_raw.columns:
        raise ValueError("Neither 'Datetime' nor 'Datatime' column exists in diff_raw")

    value_col = 'Price' if 'Price' in diff_raw.columns else diff_raw.columns[1]
    raw_dates = pd.to_datetime(diff_raw[datetime_col])
    eval_mask = (raw_dates.dt.date >= eval_start_day) & (raw_dates.dt.date <= eval_end_day)
    eval_slice = diff_raw.loc[eval_mask, [datetime_col, value_col]].reset_index(drop=True)

    diff_true_eval = eval_slice[value_col].to_numpy(dtype=np.float32)

    if len(diff_preds) != len(diff_true_eval):
        min_len = min(len(diff_preds), len(diff_true_eval))
        print(f"Warning: pred/true length mismatch ({len(diff_preds)} vs {len(diff_true_eval)}), truncating to {min_len}")
        diff_preds = diff_preds[:min_len]
        eval_slice = eval_slice.iloc[:min_len].reset_index(drop=True)
        diff_true_eval = diff_true_eval[:min_len]

    save_df = pd.DataFrame({
        datetime_col: eval_slice[datetime_col],
        'diff_true': diff_true_eval,
        'diff_pred': diff_preds,
    })
    print(weekday_sign_accuracy(diff_preds, diff_true_eval, eval_slice[datetime_col]))
    save_df.to_csv(f"{args.model_type}_{args.eval_start_day}_diff_preds.csv", index=False)
    # if args.report:
    #     output_file = "output_report.xlsx"
    #     # 使用 xlsxwriter 引擎
    #     with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    #         df_report.to_excel(writer, index=False, header=False, sheet_name='Sheet1')
    #         workbook  = writer.book
    #         worksheet = writer.sheets['Sheet1']
    #         format_float_3 = workbook.add_format({'num_format': '0.000'})
    #         format_int = workbook.add_format({'num_format': '0'})
    #         for col_idx, col_name in enumerate(df_report.columns):
    #             first_cell_val = str(df_report.iloc[0, col_idx])
    #             if "准确率" in first_cell_val:
    #                 worksheet.set_column(col_idx, col_idx, 15, format_int)
    #             else:
    #                 worksheet.set_column(col_idx, col_idx, 15, format_float_3)
