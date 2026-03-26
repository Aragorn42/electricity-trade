#from model import #timesfm2_5time, timesfm2_5, naive_avg, holiday_avg, fixed, DLinear, PatchTST, TimeMoE#, chronos2, chronos2time, chronos2holiday, YingLong, chronos2time2
from sklearn.metrics import f1_score

from utils.data_process import handle_excel, init_report_df, add_prediction_columns
import torch
from utils.dataset import PriceDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from os import path as file
import random
import chinese_calendar as chinese_holiday

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

            if args.model_type == "HolidayAvg":
                y_pred = model.forecast(
                    args, 
                    inputs_for_model,   # [batch, Seq]
                    holiday_x=x_holiday_padded,   # [batch, Seq]
                    holiday_y=y_holiday_padded    # [batch, Pred]
                )
                y_pred = y_pred.reshape(args.batchsize, args.pred_len, 1) # [Batch, pred_len, 1]
            else:
                y_pred = model.forecast(args.pred_len, inputs_for_model, args)
            # y_pred: [Batch, 24*pred_len, Num_Quantiles]
            # 如果进行了填充，需要把填充的多余部分切掉
            if current_batch_size < args.batchsize:
                y_pred = y_pred[:current_batch_size, :, :]
            y_preds.append(y_pred)
            
        # 合并所有 Batch
        y_preds = np.concatenate(y_preds, axis=0) # [Sum_Batch, 24*pred_len, Num_Quantiles]
        y_preds = y_preds[:, -24:, :] 
        
        return y_preds.reshape(-1, y_preds.shape[-1]) # [timestramp, Num_Quantiles]

def calc_sign_accuracy_workday(preds, true_values, is_holiday):
    """
    计算符号准确率 (方向预测准确率)计算非节假日is_holiday=False的样本
    preds: numpy array, 预测值
    true_values: numpy array, 真实值
    is_holiday: list of bool, 表示每个时间点是否为节假日 (False=非节假日, True=节假日)
    """
    # 确保所有输入为一维数组（避免二维输入导致的维度问题）
    p = preds.flatten()
    t = true_values.flatten()
    is_holiday_arr = np.array(is_holiday).flatten()
    
    # 确保三个数组长度一致（取最小长度）
    min_len = min(len(p), len(t), len(is_holiday_arr))
    p = p[-min_len:]
    t = t[-min_len:]
    is_holiday_arr = is_holiday_arr[-min_len:]
    
    # 创建非节假日掩码（is_holiday=False 的位置为 True）
    mask = ~is_holiday_arr  # ~ 表示逻辑非，False -> True, True -> False
    
    # 仅保留非节假日样本
    p_non_holiday = p[mask]
    t_non_holiday = t[mask]
    
    # 计算符号并比较
    sign_p = np.sign(p_non_holiday)
    sign_t = np.sign(t_non_holiday)
    
    correct_count = np.sum(sign_p == sign_t)
    accuracy = correct_count / len(p_non_holiday)
    return accuracy

def calc_workday_mae(preds, true_values, is_holiday):
    """
    计算工作日样本的 MAE（is_holiday=False 的样本）
    preds: numpy array, 预测值
    true_values: numpy array, 真实值
    is_holiday: list of bool, 表示每个时间点是否为节假日 (False=工作日, True=节假日)
    """
    # 确保所有输入为一维数组（避免二维输入导致的维度问题）
    p = preds.flatten()
    t = true_values.flatten()
    is_holiday_arr = np.array(is_holiday).flatten()

    # 确保三个数组长度一致（取最小长度）
    min_len = min(len(p), len(t), len(is_holiday_arr))
    p = p[-min_len:]
    t = t[-min_len:]
    is_holiday_arr = is_holiday_arr[-min_len:]

    # 创建工作日掩码（is_holiday=False 的位置为 True）
    mask = ~is_holiday_arr

    # 仅保留工作日样本
    p_workday = p[mask]
    t_workday = t[mask]

    if len(p_workday) == 0:
        return np.inf

    # MSE
    mse = np.mean((p_workday - t_workday) ** 2)

    # MAPE
    eps = 1e-8
    mape = np.mean(np.abs((p_workday - t_workday) / (t_workday + eps))) * 100.0

    # Huber_loss
    delta = 1.0
    err = p_workday - t_workday
    is_small = np.abs(err) <= delta
    huber_elem = np.where(is_small, 0.5 * err ** 2, delta * (np.abs(err) - 0.5 * delta))
    huber_loss = np.mean(huber_elem)

    # MAE
    mae = np.mean(np.abs(p_workday - t_workday))
    return mae
# def calc_sign_f1_workday(preds, true_values, is_holiday, average='macro'):
#     """
#     计算符号预测的 F1-score (仅计算非节假日样本)
#     preds: numpy array, 预测值
#     true_values: numpy array, 真实值
#     is_holiday: list of bool, 表示每个时间点是否为节假日
#     average: 'macro' (兼顾正负样本), 'binary' (可指定专门看负样本)
#     """
#     # 确保所有输入为一维数组
#     p = preds.flatten()
#     t = true_values.flatten()
#     is_holiday_arr = np.array(is_holiday).flatten()
    
#     # 确保三个数组长度一致（取最小长度）
#     min_len = min(len(p), len(t), len(is_holiday_arr))
#     p = p[-min_len:]
#     t = t[-min_len:]
#     is_holiday_arr = is_holiday_arr[-min_len:]
    
#     # 创建非节假日掩码
#     mask = ~is_holiday_arr  
    
#     # 仅保留非节假日样本
#     p_non_holiday = p[mask]
#     t_non_holiday = t[mask]
    
#     if len(p_non_holiday) == 0:
#         return 0.0  # 防止除零错误

#     # 将连续数值转换为类别标签：正数为 1，负数和零为 -1
#     # 这样就把回归/数值问题变成了二分类问题
#     pred_class = np.where(p_non_holiday > 0, 1, -1)
#     true_class = np.where(t_non_holiday > 0, 1, -1)
    
#     # 计算 F1-score
#     # 'macro' 意味着模型必须在正电价和负电价上都预测得准，得分才会高
#     score = f1_score(true_class, pred_class, average=average)
    
#     return score
    

def evaluate(args):
    # df with datatime and Price
    _, _, diff_raw = handle_excel(args.file_path)
    diff, diff_scaler = scaled_data(diff_raw)
    if file.exists("output_report.xlsx"):
        df_report = pd.read_excel("output_report.xlsx", header=None)
    else:
        df_report = init_report_df(args, diff_raw)
    device = torch.device('cuda')

    model = get_model(args)
    diff_preds = forecast(model, diff, args) # [timestamp, Num_Quantiles]
    
    orig_shape = diff_preds.shape
    diff_preds = diff_scaler.inverse_transform(
        diff_preds.reshape(-1, 1)
    ).reshape(orig_shape)
    
    diff_true = diff_raw.iloc[1:, 1].values
    timestramp = diff_raw.iloc[-(args.eval_day*24):, 0].values
    # if args.find_best_quants:
    if args.find_quant:
        diff_preds = get_best_quants(
            timestramp,
            diff_preds,
            diff_true,
            start1=getattr(args, "quant_start_day1", None),
            end1=getattr(args, "quant_end_day1", None),
            start2=getattr(args, "quant_start_day2", None),
            end2=getattr(args, "quant_end_day2", None)
        ) # [timestamp]
    else:
        diff_preds = diff_preds[:, int(len(diff_preds[0])*0.6) if args.model_type != "YingLongtime" else (int(len(diff_preds[0])*0.8))]
    df_report = add_prediction_columns(df_report, diff_preds, diff_true, args, is_two_variate=False)
    if args.model_type == "Chronos-2time2":
        df_report.to_excel(f"chronos2time2{args.quant_start_day1}.xlsx", index=False, header=False)
    if args.report:
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

def get_best_quants(timestramp, diff_preds, diff_true, start1=None, end1=None, start2=None, end2=None):
    # diff_preds shape -> [timestamp, Num_Quantiles]
    preds = np.asarray(diff_preds)

    true_arr = np.asarray(diff_true).flatten()
    times = pd.to_datetime(np.asarray(timestramp))

    min_len = min(len(times), len(preds), len(true_arr))
    if min_len == 0:
        return np.array([])

    times = times[-min_len:]
    preds = preds[-min_len:]
    true_arr = true_arr[-min_len:]

    valid_time_mask = ~pd.isna(times)
    range_mask = valid_time_mask.copy()

    if start1 is not None:
        start_dt = pd.to_datetime(start1).normalize()
        range_mask &= (times >= start_dt)
    if end1 is not None:
        end_dt = pd.to_datetime(end1).normalize() + pd.Timedelta(hours=23)
        range_mask &= (times <= end_dt)
    if start2 is not None:
        start_dt2 = pd.to_datetime(start2).normalize()
        range_mask &= (times >= start_dt2)
    if end2 is not None:
        end_dt2 = pd.to_datetime(end2).normalize() + pd.Timedelta(hours=23)
        range_mask &= (times <= end_dt2)
        
    if not np.any(range_mask):
        range_mask = valid_time_mask

    if not np.any(range_mask):
        return preds[:, 0]

    times_in_range = times[range_mask]
    is_workday_in_range = np.array([
        chinese_holiday.is_workday(ts.date()) if pd.notna(ts) else False
        for ts in times_in_range
    ], dtype=bool)
    is_holiday_in_range = ~is_workday_in_range
    
    num_quants = preds.shape[1]
    init_q = int(num_quants * 0.6)

    # 初始基线：全小时都用同一个分位
    best_preds = preds[:, init_q].copy()
    selected_quants = np.full(24, init_q, dtype=int)

    current_best_mae = calc_workday_mae(
        best_preds[range_mask],
        true_arr[range_mask],
        is_holiday_in_range
    )

    time_hours = pd.Series(times).dt.hour.to_numpy()

    for hour in range(24):
        hour_mask = (time_hours == hour)

        best_hour_quant = selected_quants[hour]
        improved = False

        for quant_idx in range(num_quants):
            if quant_idx == selected_quants[hour]:
                continue

            candidate_preds = best_preds.copy()
            candidate_preds[hour_mask] = preds[hour_mask, quant_idx]

            mae = calc_workday_mae(
                candidate_preds[range_mask],
                true_arr[range_mask],
                is_holiday_in_range
            )

            if mae < current_best_mae:
                current_best_mae = mae
                best_hour_quant = quant_idx
                improved = True

        selected_quants[hour] = best_hour_quant
        best_preds[hour_mask] = preds[hour_mask, best_hour_quant]
    final_mae = calc_workday_mae(
        best_preds[range_mask],
        true_arr[range_mask],
        is_holiday_in_range
    )
    print(f"Greedy quantile indices by hour: {selected_quants.tolist()}")
    print(f"Best workday MAE in range: {final_mae:.4f}")

    return best_preds