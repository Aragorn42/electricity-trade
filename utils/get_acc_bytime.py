import argparse
import time

import numpy as np
import pandas as pd
from chinese_calendar import is_workday


def transform_excel_structure(file_path):
    df = pd.read_excel(file_path)
    y_true = (df["日前实时差价"] > 0).astype(int)
    print("已提取真实标签 (基于 '日前实时差价' > 0)")

    # 创建新的 DataFrame，先保留前 2 列
    new_df = df.iloc[:, [0, 1]].copy()

    model_acc_cols = [c for c in df.columns if "准确率" in c]
    print(f"检测到 {len(model_acc_cols)} 个模型列")

    for col in model_acc_cols:
        # 如果原列值==1(判断正确)，则预测值==真实值；否则预测值==1-真实值
        model_pred = np.where(df[col] == 1, y_true, 1 - y_true)
        clean_name = col.split("_")[0].strip()
        new_df[f"{clean_name}_预测结果"] = model_pred
        new_df[col] = df[col]

    # new_df.to_excel("temp.xlsx", index=False)
    return new_df


def vote(input_path):
    df = transform_excel_structure(input_path)
    df["时间戳"] = pd.to_datetime(df["时间戳"])

    pred_cols = [c for c in df.columns if "预测结果" in c]
    tmp = df[pred_cols].apply(pd.to_numeric, errors="coerce").astype(int)
    model_vote_bits = tmp.astype(str).agg("".join, axis=1)
    ones = tmp.sum(axis=1)
    result = (ones > (len(pred_cols) - ones)).astype(int)
    out = pd.DataFrame({"时间戳": df["时间戳"], "预测结果": result, "模型判断结果": model_vote_bits})
    #out.to_excel(f"{time.strftime('%Y%m%d_%H%M%S')}_vote.xlsx", index=False)
    return out


def get_accuracy_table(file_path, start_date, end_date, recognize):
    df = vote(file_path)

    raw_df = pd.read_excel(file_path)
    raw_df["时间戳"] = pd.to_datetime(raw_df["时间戳"])
    raw_df["日前实时差价"] = pd.to_numeric(raw_df["日前实时差价"], errors="coerce")
    raw_df["真实符号"] = (raw_df["日前实时差价"] > 0).astype(int)

    df["时间戳"] = pd.to_datetime(df["时间戳"])
    df = df.merge(raw_df[["时间戳", "真实符号"]], on="时间戳", how="left")

    df["小时"] = df["时间戳"].dt.hour
    df["类型"] = df["时间戳"].dt.date.apply(lambda x: "工作日" if is_workday(x) else "非工作日")

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(hours=23)
    df = df[(df["时间戳"] >= start_ts) & (df["时间戳"] <= end_ts)]
    data_cols = [col for col in df.columns if "预测结果" in col]

    df[data_cols] = df[data_cols].apply(pd.to_numeric, errors="coerce").astype(int)

    for col in data_cols:
        df[col] = (df[col] == df["真实符号"]).astype(int)

        table = df.groupby(["类型", "小时"])[col].mean().unstack(fill_value=0)
        table = table.reindex(index=["工作日", "非工作日"], columns=range(24))
        print("检测到的列名:", data_cols)

        table["整体准确率"] = table.mean(axis=1)
        table = table * 100
        table = table.round(2)

        col_name = df.columns[df.columns.get_loc(col) - 1]
        print(f"{col_name} 准确率表")
        print(table)

        #table.to_excel(f"{recognize}_{start_date}.xlsx")


def main():
    parser = argparse.ArgumentParser(description="Compute grouped hourly accuracy tables.")
    parser.add_argument("--file_path", default="output_report.xlsx", help="Input Excel file path")
    parser.add_argument("--start_date", default="2025-04-01", help="Start date, e.g. 2026-01-01")
    parser.add_argument("--end_date", default="2025-04-30", help="End date, e.g. 2026-01-31")
    parser.add_argument("--recognize", default="output", help="Output file prefix")
    args = parser.parse_args()

    get_accuracy_table(
        file_path=args.file_path,
        start_date=args.start_date,
        end_date=args.end_date,
        recognize=args.recognize,
    )


if __name__ == "__main__":
    main()
