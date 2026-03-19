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

# 历史7个月
# export quant_start_day1=2025-06-01
# export quant_end_day1=2025-12-31
# "$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# run_accuracy 2026-01-01 2026-01-31 "历史7个月5模型"
# run_accuracy 2026-02-01 2026-02-28 "历史7个月5模型"
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# run_accuracy 2026-01-01 2026-01-31 "历史7个月9模型"
# run_accuracy 2026-02-01 2026-02-28 "历史7个月9模型"
# rm -rf "$REPORT_FILE"

#历史3个月1月
# export quant_start_day1=2025-10-01
# export quant_end_day1=2025-12-31
# "$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# run_accuracy 2026-01-01 2026-01-31 "历史3个月5模型"
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# run_accuracy 2026-01-01 2026-01-31 "历史3个月9模型"
# rm -rf "$REPORT_FILE"

# # 历史3个月2月
# export quant_start_day1=2025-10-01
# export quant_end_day1=2025-12-31
# "$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# run_accuracy 2026-02-01 2026-02-28 "历史3个月5模型"
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# run_accuracy 2026-02-01 2026-02-28 "历史3个月9模型"
# rm -rf "$REPORT_FILE"

# # 历史1个月1月
# export quant_start_day1=2025-12-01
# export quant_end_day1=2025-12-31
# "$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# run_accuracy 2026-01-01 2026-01-31 "历史1个月5模型"
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# run_accuracy 2026-01-01 2026-01-31 "历史1个月9模型"
# rm -rf "$REPORT_FILE"

# # # 历史1个月2月
# export quant_start_day1=2026-01-01
# export quant_end_day1=2026-01-31
# "$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant
# run_accuracy 2026-02-01 2026-02-28 "历史1个月5模型"
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report
# run_accuracy 2026-02-01 2026-02-28 "历史1个月9模型"
# rm -rf "$REPORT_FILE"

# # 环比1个月1月
# export quant_start_day1=2025-01-01
# export quant_end_day1=2025-01-31
# "$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=450
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant --eval_day=450
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant --eval_day=450
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant --eval_day=450
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant --eval_day=450
# run_accuracy 2026-01-01 2026-01-31 "环比1个月5模型"
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --eval_day=450
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --eval_day=450
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --eval_day=450
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --eval_day=450
# run_accuracy 2026-01-01 2026-01-31 "环比1个月9模型"
# rm -rf "$REPORT_FILE"

# # 环比1个月2月
# export quant_start_day1=2025-02-01
# export quant_end_day1=2025-02-28
# "$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=450
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant --eval_day=450
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant --eval_day=450
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant --eval_day=450
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --find_quant --eval_day=450
# run_accuracy 2026-02-01 2026-02-28 "环比1个月5模型"
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --eval_day=450
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --eval_day=450
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --eval_day=450
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --report --eval_day=450
# run_accuracy 2026-02-01 2026-02-28 "环比1个月9模型"
# rm -rf "$REPORT_FILE"


# # 环比1个月+历史1个月1月
# export quant_start_day1=2025-01-01
# export quant_end_day1=2025-01-31
# export quant_start_day2=2025-12-01
# export quant_end_day2=2025-12-31
# "$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=450
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
# run_accuracy 2026-01-01 2026-01-31 "环比1个月+历史1个月5模型"
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450  
# run_accuracy 2026-01-01 2026-01-31 "环比1个月+历史1个月9模型"
# rm -rf "$REPORT_FILE"

# # 环比1个月+历史3个月1月
# export quant_start_day1=2025-01-01
# export quant_end_day1=2025-01-31
# export quant_start_day2=2025-10-01
# export quant_end_day2=2025-12-31
# "$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=450
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
# run_accuracy 2026-01-01 2026-01-31 "环比1个月+历史3个月5模型"
# "$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450
# "$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450
# "$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450
# "$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450  
# run_accuracy 2026-01-01 2026-01-31 "环比1个月+历史3个月9模型"
# rm -rf "$REPORT_FILE"

# 环比1个月+历史1个月2月
export quant_start_day1=2025-02-01
export quant_end_day1=2025-02-28
export quant_start_day2=2026-01-01
export quant_end_day2=2026-01-31
"$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=450
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
run_accuracy 2026-02-01 2026-02-28 "环比1个月+历史1个月5模型"
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1 " --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2 " --report --eval_day=450
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450  
run_accuracy 2026-02-01 2026-02-28 "环比1个月+历史1个月9模型"
rm -rf "$REPORT_FILE"

# 环比1个月+历史3个月2月
export quant_start_day1=2025-02-01
export quant_end_day1=2025-02-28
export quant_start_day2=2025-11-01
export quant_end_day2=2026-01-31
"$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=450
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=450
run_accuracy 2026-02-01 2026-02-28 "环比1个月+历史3个月5模型"
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day1="$quant_start_day1" --quant_end_day1="$quant_end_day1" --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=450  
run_accuracy 2026-02-01 2026-02-28 "环比1个月+历史3个月9模型"
rm -rf "$REPORT_FILE"
