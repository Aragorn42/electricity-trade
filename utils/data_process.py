'''
from xlsm to 3 csv
'''

import pandas as pd
import numpy as np

def init_report_df(args, da_file, rt_file, diff_file):
    df1 = da_file.iloc[-(args.eval_day*24):, :]
    df2 = rt_file.iloc[-(args.eval_day*24):, :]
    df3 = diff_file.iloc[-(args.eval_day*24):, :]
    
    df1 = df1.rename(columns={'Price': 'DA_Price'})
    df2 = df2.rename(columns={'Price': 'RT_Price'})
    df3 = df3.rename(columns={'Price': 'Diff_Price'})
    
    merged_df = pd.merge(df1, df2, on='Datetime', how='outer')
    merged_df = pd.merge(merged_df, df3, on='Datetime', how='outer')
    
    target_cols = ['Datetime', 'DA_Price', 'RT_Price', 'Diff_Price']
    merged_df = merged_df[target_cols]
    
    merged_df = merged_df.sort_values('Datetime').reset_index(drop=True)

    header_data = [
        ['时间戳', '日前电价', '实时电价', '日前实时差价']
    ]
    
    header_df = pd.DataFrame(header_data, columns=target_cols)
    
    # 6. 垂直拼接
    final_df = pd.concat([header_df, merged_df], ignore_index=True)
    
    # 转为 object 类型，方便后续写入字符串
    final_df = final_df.astype(object)
    
    return final_df

def add_prediction_columns(df, pred_array, true_values, args, is_two_variate = False):
    if hasattr(pred_array, 'detach'):
        pred_array = pred_array.detach().cpu().numpy()
    if hasattr(pred_array, 'values'):
        pred_array = pred_array.values
    preds = np.array(pred_array).flatten()
    
    n_preds = len(preds)
    total_rows = len(df)
    if is_two_variate:
        col_pred_name = f"{args.model_type}_pred_results_two" 
    else:
        col_pred_name = f"{args.model_type}_pred_results_single"
    df[col_pred_name] = np.nan
    df[col_pred_name] = df[col_pred_name].astype(object)
    col_pred_idx = df.columns.get_loc(col_pred_name)
    # column name actually
    if is_two_variate:
        df.iloc[0, col_pred_idx] = f"{args.model_type}双预测_{args.seq_len}to{args.pred_len}"
    else:
        df.iloc[0, col_pred_idx] = f"{args.model_type}差价预测_{args.seq_len}to{args.pred_len}"
    df.iloc[-n_preds:, col_pred_idx] = preds

    if is_two_variate:
        col_acc_name = f"{args.model_type}_Acc_in{args.pred_len}_two"
    else:
        col_acc_name = f"{args.model_type}_Acc_in{args.pred_len}_single"
    # useless, dispatch when converting to excel
    df[col_acc_name] = np.nan
    df[col_acc_name] = df[col_acc_name].astype(object)
    
    col_acc_idx = df.columns.get_loc(col_acc_name)

    true_arr = np.array(true_values).flatten()
    relevant_true = true_arr[-n_preds:]
    results = (np.sign(preds) == np.sign(relevant_true)).astype(int)
    df.iloc[-n_preds:, col_acc_idx] = results
    
    acc = np.mean(results) * 100
    print(f"{args.model_type}{args.seq_len}to120 准确率: {acc:.2f}%")
    df.iloc[0, col_acc_idx] = f"准确率={acc:.2f}%"
    
    return df

def handle_excel(file_path):
    # header=0 表示第一行是表头(时间点)
    # usecols=range(25) 表示只读取前25列 (第1列日期 + 24列电价)，自动忽略后面的备注列
    ans = []
    for sheet_index in range(0, 3):
        
        df = pd.read_excel(file_path, sheet_name=sheet_index, engine='openpyxl', header=0, usecols=range(25))
        # 假设第一列的名字不固定，我们直接用 iloc 获取第一列的列名
        date_col_name = df.columns[0]
        df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
        df = df.dropna(subset=[date_col_name])
        # 3. 数据重塑 (Unpivot / Melt)
        # 将宽表变成长表
        # id_vars: 保持不变的列（日期）
        # var_name: 原来的列头变成的新列名（这里是“小时”）
        # value_name: 单元格里的值变成的新列名（这里是“电价”）
        melted_df = df.melt(id_vars=[date_col_name], var_name='Hour_Raw', value_name='Price')

        
        def parse_hour(val):
            """尝试从列头解析出小时数 (int)"""
            # 如果已经是数字
            if isinstance(val, (int, float)):
                return int(val)
            # 如果是字符串 (例如 '00:00' 或 '1点')
            txt = str(val)
            # 提取字符串中的数字，取第一个
            import re
            nums = re.findall(r'\d+', txt)
            if nums:
                return int(nums[0])
            return 0 # 兜底，如果没有数字则默认为0点

        # 应用解析函数
        melted_df['Hour_Offset'] = melted_df['Hour_Raw'].apply(parse_hour)

        # 5. 合成最终的“日期+时间点”列
        # 注意：有些市场定义24:00为当天的结束，但在代码中通常处理为次日00:00，
        # 这里默认假设列头代表起始小时（例如0代表00:00-01:00）
        melted_df['Datetime'] = melted_df[date_col_name] + pd.to_timedelta(melted_df['Hour_Offset'], unit='h')

        # 6. 整理最终表格
        # 只保留两列：[日期时间, 电价]
        final_df = melted_df[['Datetime', 'Price']].sort_values(by='Datetime')
        ans.append(final_df)

    return (ans[0], ans[1], ans[2])