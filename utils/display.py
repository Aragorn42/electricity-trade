import pandas as pd
from chinese_calendar import is_workday
from vote import vote
# 1. 读取 Excel 文件 (请替换为实际文件路径)
def main(strategy='vote', eval_month=1):
    if eval_month == 1:
        df = vote(strategy=strategy, input_file='output_report.xlsx', start_time='2025-06-01', end_time='2025-12-31', eval_start_time='2026-01-01', eval_end_time='2026-01-31')
    elif eval_month == 2:
        df = vote(strategy=strategy, input_file='output_report.xlsx', start_time='2025-06-01', end_time='2025-12-31', eval_start_time='2026-02-01', eval_end_time='2026-02-28')
    # 2. 时间戳处理：提取小时和类型（工作日/非工作日）
    df['时间戳'] = pd.to_datetime(df['时间戳'])
    df['小时'] = df['时间戳'].dt.hour
    df['类型'] = df['时间戳'].dt.date.apply(lambda x: '工作日' if is_workday(x) else '非工作日')


    data_cols = [col for col in df.columns if 'is_correct' in col]

    # 4. 将数据列强制转换为数值类型（0/1）
    df[data_cols] = df[data_cols].apply(pd.to_numeric, errors='coerce')

    for col in data_cols:
        table = df.groupby(['类型', '小时'])[col].mean().unstack(fill_value=0)
        table = table.reindex(index=['工作日', '非工作日'], columns=range(24))
        table['整体准确率'] = table.mean(axis=1)

        table = table * 100
        table = table.round(2)
        print(table)
        table.to_excel(f"{eval_month}月{strategy}_accuracy_table.xlsx")

if __name__ == "__main__":
    main(strategy='vote', eval_month=1)
    main(strategy='workday', eval_month=2)
    