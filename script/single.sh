#!/usr/bin/env bash
set -euo pipefail

PY_BIN="/home/liym/miniconda3/bin/python"
CONDA_BIN="/home/liym/miniconda3/bin/conda"
HELPER_SCRIPT="./utils/get_acc_bytime.py"
REPORT_FILE="./output_report.xlsx"

run_accuracy() {
  local start_date="$1"
  local end_date="$2"
  local recognize="$3"

  "$PY_BIN" "$HELPER_SCRIPT" \
    --file_path "$REPORT_FILE" \
    --start_date "$start_date" \
    --end_date "$end_date" \
    --recognize "$recognize"
}

# 环比1月历史3月2025.4
export val_start_day=2024-04-01
export val_end_day=2024-04-30
export eval_start_day=2025-04-01
export eval_end_day=2025-04-30
export train_start_day=2025-01-01
export train_end_day=2025-03-31

"$PY_BIN" -u main.py --model_type=DLinear --seq_len=336 --val_start_day="$val_start_day" --val_end_day="$val_end_day" --eval_start_day="$eval_start_day" --eval_end_day="$eval_end_day" --train_start_day="$train_start_day" --train_end_day="$train_end_day" --report --need_train

# run_accuracy 2025-04-01 2025-04-30 "环比1个月+历史3个月5模型"
rm -rf "$REPORT_FILE"

# 环比1月历史3月2025.5
export val_start_day=2024-05-01
export val_end_day=2024-05-31
export eval_start_day=2025-05-01
export eval_end_day=2025-05-31
export train_start_day=2025-02-01
export train_end_day=2025-04-30

"$PY_BIN" -u main.py --model_type=DLinear --seq_len=336 --val_start_day="$val_start_day" --val_end_day="$val_end_day" --eval_start_day="$eval_start_day" --eval_end_day="$eval_end_day" --train_start_day="$train_start_day" --train_end_day="$train_end_day" --report --need_train

# run_accuracy 2025-05-01 2025-05-31 "环比1个月+历史3个月5模型"
rm -rf "$REPORT_FILE"
# 环比1月历史3月2025.6
export val_start_day=2024-06-01
export val_end_day=2024-06-30
export eval_start_day=2025-06-01
export eval_end_day=2025-06-30
export train_start_day=2025-03-01
export train_end_day=2025-05-31

"$PY_BIN" -u main.py --model_type=DLinear --seq_len=336 --val_start_day="$val_start_day" --val_end_day="$val_end_day" --eval_start_day="$eval_start_day" --eval_end_day="$eval_end_day" --train_start_day="$train_start_day" --train_end_day="$train_end_day" --report --need_train

# run_accuracy 2025-06-01 2025-06-30 "环比1个月+历史3个月5模型"
rm -rf "$REPORT_FILE"

# 环比1月历史3月2025.7
export val_start_day=2024-07-01
export val_end_day=2024-07-31
export eval_start_day=2025-07-01
export eval_end_day=2025-07-31
export train_start_day=2025-04-01
export train_end_day=2025-06-30

"$PY_BIN" -u main.py --model_type=DLinear --seq_len=336 --val_start_day="$val_start_day" --val_end_day="$val_end_day" --eval_start_day="$eval_start_day" --eval_end_day="$eval_end_day" --train_start_day="$train_start_day" --train_end_day="$train_end_day" --report --need_train

# run_accuracy 2025-07-01 2025-07-31 "环比1个月+历史3个月5模型"
rm -rf "$REPORT_FILE"

# 环比1月历史3月2025.8
export val_start_day=2024-08-01
export val_end_day=2024-08-31
export eval_start_day=2025-08-01
export eval_end_day=2025-08-31
export train_start_day=2025-05-01
export train_end_day=2025-07-31

"$PY_BIN" -u main.py --model_type=DLinear --seq_len=336 --val_start_day="$val_start_day" --val_end_day="$val_end_day" --eval_start_day="$eval_start_day" --eval_end_day="$eval_end_day" --train_start_day="$train_start_day" --train_end_day="$train_end_day" --report --need_train

# run_accuracy 2025-08-01 2025-08-31 "环比1个月+历史3个月5模型"
rm -rf "$REPORT_FILE"

# 环比1月历史3月2025.9
export val_start_day=2024-09-01
export val_end_day=2024-09-30
export eval_start_day=2025-09-01
export eval_end_day=2025-09-30
export train_start_day=2025-06-01
export train_end_day=2025-08-31

"$PY_BIN" -u main.py --model_type=DLinear --seq_len=336 --val_start_day="$val_start_day" --val_end_day="$val_end_day" --eval_start_day="$eval_start_day" --eval_end_day="$eval_end_day" --train_start_day="$train_start_day" --train_end_day="$train_end_day" --report --need_train

# run_accuracy 2025-09-01 2025-09-30 "环比1个月+历史3个月5模型"
rm -rf "$REPORT_FILE"

# 环比1月历史3月2025.10
export val_start_day=2024-10-01
export val_end_day=2024-10-31
export eval_start_day=2025-10-01
export eval_end_day=2025-10-31
export train_start_day=2025-07-01
export train_end_day=2025-09-30

"$PY_BIN" -u main.py --model_type=DLinear --seq_len=336 --val_start_day="$val_start_day" --val_end_day="$val_end_day" --eval_start_day="$eval_start_day" --eval_end_day="$eval_end_day" --train_start_day="$train_start_day" --train_end_day="$train_end_day" --report --need_train

# run_accuracy 2025-10-01 2025-10-31 "环比1个月+历史3个月5模型"
rm -rf "$REPORT_FILE"

# 环比1月历史3月2025.11
export val_start_day=2024-11-01
export val_end_day=2024-11-30
export eval_start_day=2025-11-01
export eval_end_day=2025-11-30
export train_start_day=2025-08-01
export train_end_day=2025-10-31

"$PY_BIN" -u main.py --model_type=DLinear --seq_len=336 --val_start_day="$val_start_day" --val_end_day="$val_end_day" --eval_start_day="$eval_start_day" --eval_end_day="$eval_end_day" --train_start_day="$train_start_day" --train_end_day="$train_end_day" --report --need_train

# run_accuracy 2025-11-01 2025-11-30 "环比1个月+历史3个月5模型"
rm -rf "$REPORT_FILE"

# 环比1月历史3月2025.12
export val_start_day=2024-12-01
export val_end_day=2024-12-31
export eval_start_day=2025-12-01
export eval_end_day=2025-12-31
export train_start_day=2025-09-01
export train_end_day=2025-11-30

"$PY_BIN" -u main.py --model_type=DLinear --seq_len=336 --val_start_day="$val_start_day" --val_end_day="$val_end_day" --eval_start_day="$eval_start_day" --eval_end_day="$eval_end_day" --train_start_day="$train_start_day" --train_end_day="$train_end_day" --report --need_train

# run_accuracy 2025-12-01 2025-12-31 "环比1个月+历史3个月5模型"
rm -rf "$REPORT_FILE"

# # 环比1个月+历史3个月1月
export val_start_day=2025-01-01
export val_end_day=2025-01-31
export eval_start_day=2026-01-01
export eval_end_day=2026-01-31
export train_start_day=2025-10-01
export train_end_day=2025-12-31

"$PY_BIN" -u main.py --model_type=DLinear --seq_len=336 --val_start_day="$val_start_day" --val_end_day="$val_end_day" --eval_start_day="$eval_start_day" --eval_end_day="$eval_end_day" --train_start_day="$train_start_day" --train_end_day="$train_end_day" --report --need_train

# run_accuracy 2026-01-01 2026-01-31 "环比1个月+历史3个月5模型"
rm -rf "$REPORT_FILE"

# 环比1个月+历史3个月2月
export val_start_day=2025-02-01
export val_end_day=2025-02-28
export eval_start_day=2026-02-01
export eval_end_day=2026-02-28
export train_start_day=2025-11-01
export train_end_day=2026-01-31

"$PY_BIN" -u main.py --model_type=DLinear --seq_len=336 --val_start_day="$val_start_day" --val_end_day="$val_end_day" --eval_start_day="$eval_start_day" --eval_end_day="$eval_end_day" --train_start_day="$train_start_day" --train_end_day="$train_end_day" --report --need_train

# run_accuracy 2026-02-01 2026-02-28 "环比1个月+历史3个月5模型"
rm -rf "$REPORT_FILE"