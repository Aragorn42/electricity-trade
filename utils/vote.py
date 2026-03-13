import pandas as pd
import numpy as np
import chinese_calendar


def transform_excel_structure(file_path):
    
    df = pd.read_excel(file_path)
    y_true = (df['日前实时差价'] > 0).astype(int)
    print("已提取真实标签 (基于 '日前实时差价' > 0)")

    # 创建新的 DataFrame，先保留前4列 (索引0,1,2,3)
    new_df = df.iloc[:, 0:4].copy()

    model_acc_cols = [c for c in df.columns if "准确率" in c]
    print(f"检测到 {len(model_acc_cols)} 个模型列")
    df_columns = df.columns
    for col in model_acc_cols:
        # 如果 原列==1 (判断正确)，则 预测值 == 真实值
        # 如果 原列==0 (判断错误)，则 预测值 == (1 - 真实值)
        model_pred = np.where(df[col] == 1, y_true, 1 - y_true)
        
        # 提取模型名称作为列名
        current_idx = df_columns.get_loc(col)
        clean_name = df_columns[current_idx - 1]
        clean_name = col.split('_')[0].strip()
        new_df[f'{clean_name}_预测结果'] = model_pred
        new_df[col] = df[col]
        
    #new_df.to_excel(save_path, index=False)
    return new_df

def get_combo_pattern(df, model_cols, ground_truth_col='日前实时差价'):
    if ground_truth_col in df.columns:
        df['true_label'] = df[ground_truth_col].apply(lambda x: 1 if x >= 0 else 0)
    for col in model_cols:
        df[col] = df[col].astype(int)

    df['combo_pattern'] = df[model_cols].astype(str).agg(''.join, axis=1)
    return df

def get_decision_table(df, model_cols, strategy='historical', start_time=None, end_time=None):
    # - df: 包含 '时间戳', 'combo_pattern', 'true_label' 等列的数据框
    # - model_cols: 模型列名列表
    # - strategy: 策略选择
    #     'voting'     -> 简单投票
    #     'historical' -> 基于历史范围内所有数据的表现
    #     'workday'    -> 基于历史范围内 工作日 数据的表现
    # - start_time: 过滤的开始时间 (可选, 格式如 '2023-01-01')
    # - end_time: 过滤的结束时间 (可选, 格式如 '2023-12-31')

    df_filtered = df.copy()

    if start_time is not None or end_time is not None:
        df_filtered['时间戳'] = pd.to_datetime(df_filtered['时间戳'])
        if start_time is not None:
            df_filtered = df_filtered[df_filtered['时间戳'] >= pd.to_datetime(start_time)]
        if end_time is not None:
            df_filtered = df_filtered[df_filtered['时间戳'] <= pd.to_datetime(end_time)]

    if strategy == 'workday':
        is_holiday_mask = df_filtered['时间戳'].apply(
            lambda x: chinese_calendar.is_holiday(x.date()) if pd.notnull(x) else False
        )
        # 只保留范围内的工作日做出决策
        df_filtered = df_filtered[~is_holiday_mask]

    group_keys = ['combo_pattern']

    if strategy in['historical', 'workday']:
        lookup_table = df_filtered.groupby(group_keys)['true_label'].agg(['count', 'mean']).reset_index()
        lookup_table['final_strategy'] = lookup_table['mean'].apply(lambda x: 1 if x > 0.5 else 0)

    elif strategy == 'voting':
        # 简单投票策略：提取所有出现过的组合
        lookup_table = df_filtered[group_keys].drop_duplicates().copy()
        vote_threshold = len(model_cols) / 2
        lookup_table['final_strategy'] = lookup_table['combo_pattern'].apply(
            lambda x: 1 if x.count('1') > vote_threshold else 0
        )
        pattern_counts = df_filtered.groupby(group_keys).size()
        lookup_table['count'] = lookup_table['combo_pattern'].map(pattern_counts).fillna(0).astype(int)
        lookup_table['mean'] = np.nan 
        
    return lookup_table


import pandas as pd

def apply_decision(df, decision_table, model_cols, start_time=None, end_time=None):
    # 将决策表映射回数据，给出最终预测，并评估指定时间范围内的准确率。

    # - df: 包含 'combo_pattern', 'true_label', '时间戳' 等列的数据框
    # - decision_table: 训练好（获取到）的决策表
    # - model_cols: 模型列名列表
    # - start_time: 评估的开始时间 (可选, 格式如 '2023-07-01')
    # - end_time: 评估的结束时间 (可选, 格式如 '2023-12-31')
    # - df_filtered: 包含预测结果的数据框 (仅包含指定时间范围内的数据)
    # - accuracy: 在该时间范围内的准确率

    df_filtered = df.copy()
    if start_time is not None:
        df_filtered = df_filtered[df_filtered['时间戳'] >= pd.to_datetime(start_time)]
    if end_time is not None:
        df_filtered = df_filtered[df_filtered['时间戳'] <= pd.to_datetime(end_time)]
            
    if df_filtered.empty:
        print("警告：指定的时间范围内没有数据！")
        return df_filtered, 0.0

    join_keys = ['combo_pattern']
    df_filtered = pd.merge(df_filtered, decision_table[join_keys +['final_strategy']], on=join_keys, how='left')

    vote_threshold = len(model_cols) / 2
    missing_mask = df_filtered['final_strategy'].isna()
    if missing_mask.any():
        df_filtered.loc[missing_mask, 'final_strategy'] = df_filtered.loc[missing_mask, 'combo_pattern'].apply(
            lambda x: 1 if x.count('1') > vote_threshold else 0
        )
        
    df_filtered['final_strategy'] = df_filtered['final_strategy'].astype(int)
    df_filtered['is_correct'] = (df_filtered['final_strategy'] == df_filtered['true_label']).astype(int)
    accuracy = df_filtered['is_correct'].mean()

    return df_filtered, accuracy

# def maximize_accuracy_refactored(df, use_workday=False):
#     # 识别模型列
#     model_cols =[col for col in df.columns if "预测结果" in col]
#     print(f"正在分析以下模型列: {model_cols}")
#     df_processed = get_combo_pattern(df, model_cols)
    
#     table_voting = get_decision_table(df_processed, model_cols, strategy='voting', use_workday=use_workday)
#     df_voting = apply_decision(df_processed, table_voting, model_cols, use_workday=use_workday)
#     old_accuracy = df_voting['is_correct'].mean()
    
#     # 2.2 获取新策略(历史统计法)决策表，并应用
#     table_historical = get_decision_table(df_processed, model_cols, strategy='historical', use_workday=use_workday)
#     df_final = apply_decision(df_processed, table_historical, model_cols, use_workday=use_workday)
#     new_accuracy = df_final['is_correct'].mean()

#     # ===========================================================
#     # 输出详细报告
#     # ===========================================================
#     print("\n" + "="*50)
#     print("分析报告")
#     print("="*50)
#     print(f"样本总数: {len(df_final)}")
#     print(f"引入工作日区分: {'是' if use_workday else '否'}")
#     print(f"旧方法(简单投票) 准确率: {old_accuracy:.2%}")
#     print(f"新方法(查表策略) 准确率: {new_accuracy:.2%}")
#     print(f"提升幅度: +{(new_accuracy - old_accuracy):.2%}")
#     print("-" * 50)

#     print("\n关键分歧点分析 (这里决定了准确率的提升):")
#     print("说明：mean表示该组合下真实结果是1的概率。")
#     print("如果 mean > 0.5，新策略会判1；如果 mean < 0.5，新策略会判0。")
    
#     pd.set_option('display.max_rows', None)
    
#     # 筛选出有统计意义的组合 (样本数 > 10) 并按样本数降序排列
#     significant_patterns = table_historical[table_historical['count'] > 10].sort_values(by='count', ascending=False)
    
#     # 输出决策表核心信息
#     display_cols =['combo_pattern', 'count', 'mean', 'final_strategy']
#     if use_workday: display_cols.insert(1, 'is_workday')
#     print(significant_patterns[display_cols].to_string(index=False))
    
#     # 自动保存结果到新Excel
#     output_file = 'optimized_result.xlsx'
#     df_final.to_excel(output_file, index=False)
#     print(f"\n详细结果已保存至: {output_file}")
    
#     return df_final, table_historical


def vote(strategy='vote', input_file='output_report.xlsx', start_time='2025-06-01', end_time='2025-12-31', eval_start_time='2026-01-01', eval_end_time='2026-01-31'):
    gt_col_name = "日前实时差价"  # 真实标签列名
    converted_df = transform_excel_structure(input_file)
    
    model_cols =[col for col in converted_df.columns if "预测结果" in col]
    df_processed = get_combo_pattern(converted_df, model_cols)
    decision_workday = get_decision_table(df_processed, model_cols, strategy='workday', start_time=start_time, end_time=end_time)
    decision_voting = get_decision_table(df_processed, model_cols, strategy='voting', start_time=start_time, end_time=end_time)
    
    df_vote, accuracy_vote = apply_decision(df_processed, decision_voting, model_cols, eval_start_time, eval_end_time)
    df_final, accuracy_final = apply_decision(df_processed, decision_workday, model_cols, eval_start_time, eval_end_time)
    
    print(f"简单投票策略准确率: {accuracy_vote:.2%}")
    print(f"基于工作日区分的策略准确率: {accuracy_final:.2%}")
    return df_vote if strategy == 'vote' else df_final

if __name__ == "__main__":
    df_final = vote()
    df_final.to_excel(f"optimized.xlsx", index=False, columns=['时间戳', '日前实时差价', 'combo_pattern', 'final_strategy', 'is_correct'])