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

# 历史1月2025.4
export quant_start_day2=2025-03-01
export quant_end_day2=2025-03-31
"$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=609
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
run_accuracy 2025-04-01 2025-04-30 "历史1个月5模型"
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880   --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2 " --report --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609  
run_accuracy 2025-04-01 2025-04-30 "历史1个月9模型"
rm -rf "$REPORT_FILE"

# 历史1月2025.5
export quant_start_day2=2025-04-01
export quant_end_day2=2025-04-30
"$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=609
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
run_accuracy 2025-05-01 2025-05-31 "历史1个月5模型"
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880   --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2 " --report --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609  
run_accuracy 2025-05-01 2025-05-31 "历史1个月9模型"
rm -rf "$REPORT_FILE"

# 历史1月2025.6
export quant_start_day2=2025-05-01
export quant_end_day2=2025-05-31
"$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=609
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
run_accuracy 2025-06-01 2025-06-30 "历史1个月5模型"
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880   --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2 " --report --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609  
run_accuracy 2025-06-01 2025-06-30 "历史1个月9模型"
rm -rf "$REPORT_FILE"

# 历史1月2025.7
export quant_start_day2=2025-06-01
export quant_end_day2=2025-06-30
"$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=609
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
run_accuracy 2025-07-01 2025-07-31 "历史1个月5模型"
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880   --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2 " --report --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609  
run_accuracy 2025-07-01 2025-07-31 "历史1个月9模型"
rm -rf "$REPORT_FILE"

# 历史1月2025.8
export quant_start_day2=2025-07-01
export quant_end_day2=2025-07-31
"$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=609
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
run_accuracy 2025-08-01 2025-08-31 "历史1个月5模型"
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880   --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2 " --report --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609  
run_accuracy 2025-08-01 2025-08-31 "历史1个月9模型"
rm -rf "$REPORT_FILE"

# 历史1月2025.9
export quant_start_day2=2025-08-01
export quant_end_day2=2025-08-31
"$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=609
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
run_accuracy 2025-09-01 2025-09-30 "历史1个月5模型"
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880   --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2 " --report --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609  
run_accuracy 2025-09-01 2025-09-30 "历史1个月9模型"
rm -rf "$REPORT_FILE"

# 历史1月2025.10
export quant_start_day2=2025-09-01
export quant_end_day2=2025-09-30
"$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=609
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
run_accuracy 2025-10-01 2025-10-31 "历史1个月5模型"
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880   --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2 " --report --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609  
run_accuracy 2025-10-01 2025-10-31 "历史1个月9模型"
rm -rf "$REPORT_FILE"

# 历史1月2025.11
export quant_start_day2=2025-10-01
export quant_end_day2=2025-10-31
"$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=609
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
run_accuracy 2025-11-01 2025-11-30 "历史1个月5模型"
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880   --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2 " --report --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609  
run_accuracy 2025-11-01 2025-11-30 "历史1个月9模型"
rm -rf "$REPORT_FILE"

# 历史1月2025.12
export quant_start_day2=2025-11-01
export quant_end_day2=2025-11-30
"$PY_BIN" -u main.py --model_type=HolidayAvg --seq_len=720 --report --eval_day=609
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --find_quant --eval_day=609
run_accuracy 2025-12-01 2025-12-31 "历史1个月5模型"
"$CONDA_BIN" run -n moirai --no-capture-output python -u main.py --model_type=moirai2time2 --seq_len=720 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=YingLongtime --seq_len=768 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609
"$PY_BIN" -u main.py --model_type=Chronos-2time2 --seq_len=2880   --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2 " --report --eval_day=609
"$PY_BIN" -u main.py --model_type=TimesFM-2.5time --seq_len=1080 --quant_start_day2="$quant_start_day2" --quant_end_day2="$quant_end_day2" --report --eval_day=609  
run_accuracy 2025-12-01 2025-12-31 "历史1个月9模型"
rm -rf "$REPORT_FILE"

