from model import YingLongtime2
from utils.data_process import handle_excel, init_report_df, add_prediction_columns
import torch
from utils.dataset import PriceDataset
from torch.utils.data import DataLoader
from exp.train import train
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from os import path as file
from chinese_calendar import is_holiday
import copy
def get_model(args):
    if args.model_type == "TimesFM-2.5time":
        from model import timesfm2_5time2
        model = timesfm2_5time2.Model(model_path=args.model_path)
    elif args.model_type == "TimesFM-2.5":
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
        value1 = torch.tensor([
            -42.352, -16.900, -5.879, 8.722, 19.356,16.879, -13.520, -4.291, -22.837, -6.047, 19.049, 6.494, 30.489, -9.075, 21.705, 11.469, 2.213, -1.240, -6.075, -15.271, -23.021, -26.439, -22.399, 4.697
        ])
        # 2024.1.1 to 2025.5.31, diff average value
        value2 = torch.tensor([
            0.339, 0.181, 0.250, 0.066, 0.139, 0.329, 0.211, -0.172, 0.000, 0.000, -0.003, -0.581, -1.096, -0.426, -0.024, -0.979, -0.787, -1.188, -0.324, 0.368, 0.349, -0.191, -0.152, -0.190
        ])
        model = fixed.Model(value1)
    elif args.model_type == "DLinear":
        from model import DLinear
        model = DLinear.Model(args)
    elif args.model_type == "PatchTST":
        from model import PatchTST
        model = PatchTST.Model(args)
    elif args.model_type == "Chronos-2":
        from model import chronos2
        model = chronos2.Model(args)
    elif args.model_type == "Chronos-2time":
        from model import chronos2time
        model = chronos2time.Model(args)
    elif args.model_type == "Chronos-2holiday":
        from model import chronos2holiday
        model = chronos2holiday.Model(args)
    elif args.model_type == "YingLong":
        model = YingLongtime2.Model(args)
    elif args.model_type == "moirai":
        from model import moirai
        model = moirai.Model()
    return model

def scaled_data(df):
    scaler = StandardScaler()
    df_scaled = df.copy()
    # 提取价格列(假设为第2列, index=1)并reshape为(N, 1)
    prices = df.iloc[:, 1].values.reshape(-1, 1)
    # 归一化并放回dataframe
    df_scaled.iloc[:, 1] = scaler.fit_transform(prices).flatten()
    return df_scaled, scaler
    
def forecast(model, df, args, quants=None):
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

            if args.model_type == "HolidayAvg" or args.model_type == "Chronos-2holiday":
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
                y_pred = model.forecast(args.pred_len, inputs_for_model, args, quants)
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
        return y_preds_ordered, [1.0 if is_holiday(d) else 0.0 for d in pd.to_datetime(da_dates)]
    
def calc_sign_accuracy(preds, true_values):
    """
    计算符号准确率 (方向预测准确率)
    preds: numpy array, 预测值
    true_values: numpy array, 真实值
    """
    # 展平以确保维度一致
    p = preds.flatten()
    t = true_values.flatten()
    
    # 截取较短的长度进行比较（防止长度不一致报错）
    min_len = min(len(p), len(t))
    p = p[-min_len:]
    t = t[-min_len:]
    
    # 计算符号: 1为正, -1为负, 0为0
    # 注意：在电力差价中，0的处理可能需要根据业务逻辑，这里简单认为是相等的符号
    sign_p = np.sign(p)
    sign_t = np.sign(t)

    correct_count = np.sum(sign_p == sign_t)
    accuracy = correct_count / min_len
    return accuracy

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
    
    # 检查是否有非节假日样本
    if len(p_non_holiday) == 0:
        return 0.0  # 无非节假日样本时返回0（避免除零错误）
    
    # 计算符号并比较
    sign_p = np.sign(p_non_holiday)
    sign_t = np.sign(t_non_holiday)
    
    correct_count = np.sum(sign_p == sign_t)
    accuracy = correct_count / len(p_non_holiday)
    return accuracy

def find_best_quants(model, inputs, scaler, true_values, args):
    """
    贪心算法寻找最优 Quant 组合
    """
    print("开始寻找最优分位数组合 (Greedy Search)...")
    
    # 初始化: 所有小时默认使用 quant = 12 (中位数附近)
    # 范围 0-20, 其中 12 是初始值
    best_quants = [int(args.quant_range * 0.6)] * 24
    
    baseline_preds_tensor, is_holiday = forecast(model, inputs, args, quants=best_quants)
    baseline_preds = scaler.inverse_transform(baseline_preds_tensor.reshape(-1, 1)).flatten()
    current_best_acc = calc_sign_accuracy_workday(baseline_preds, true_values, is_holiday)
    print(f"初始基准准确率 (All 12): {current_best_acc:.4f}")

    # 遍历 24 个小时
    for h in range(24):
        best_q_for_h = best_quants[h] # 默认为当前的最优值
        improved = False
        
        # 遍历该小时可能的 quant (0-20)
        # 使用 tqdm 显示进度，因为模型预测可能较慢
        for q in tqdm(range(0, args.quant_range), desc=f"Optimizing Hour {h}", leave=False):
            if q == best_quants[h]:
                continue # 跳过当前已经计算过的基准值
            
            # 构造临时的 quants 数组
            temp_quants = copy.deepcopy(best_quants)
            temp_quants[h] = q 
            
            # 预测
            preds_tensor = forecast(model, inputs, args, quants=temp_quants)
            
            # 反归一化
            preds = scaler.inverse_transform(preds_tensor.reshape(-1, 1)).flatten()
            
            # 计算准确率
            acc = calc_sign_accuracy(preds, true_values)
            
            # 贪心策略: 如果发现更好的准确率，立即更新
            # 或者是 >= 以便在准确率相同时优先选择某种偏好(这里严格 >)
            if acc > current_best_acc:
                current_best_acc = acc
                best_q_for_h = q
                improved = True
        
        # 锁定该小时的最优 quant
        best_quants[h] = best_q_for_h
        
        if improved:
            print(f"Hour {h}: 发现更优 Quant {best_q_for_h}, 当前准确率提升至: {current_best_acc:.4f}")
        else:
            print(f"Hour {h}: 保持默认 Quant {best_q_for_h}, 准确率: {current_best_acc:.4f}")

    print(f"搜索结束. 最终最优准确率: {current_best_acc:.4f}")
    print(f"最优 Quants 组合: {best_quants}")
    return best_quants

def evaluate(args):
    _, _, diff_raw = handle_excel(args.file_path)
    diff, diff_scaler = scaled_data(diff_raw) 
    true_values = diff_raw.iloc[1:, 1].values 
    model = get_model(args)
    optimal_quants = find_best_quants(model, diff, diff_scaler, true_values, args)
    

    final_preds_tensor = forecast(model, diff, args, quants=optimal_quants)
    diff_preds = diff_scaler.inverse_transform(final_preds_tensor.cpu().numpy().reshape(-1, 1)).flatten()
    
    # 再次验证准确率
    final_acc = calc_sign_accuracy(diff_preds, true_values)
    print(f"全时段准确率: {final_acc:.4f}")
    final_acc_workday = calc_sign_accuracy_workday(diff_preds, true_values)
    print(f"工作日准确率: {final_acc_workday:.4f}")

    
    return diff_preds, optimal_quants