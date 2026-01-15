from exp.eval import evaluate

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='fixed', help='Model type: 200M, HolidayAvg, NaiveAvg, 200MTime')
parser.add_argument('--model_path', type=str, default="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/TimesFM2_5_200M")
parser.add_argument('--file_path', type=str, default='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm', help='Path to the Excel file')
parser.add_argument('--seq_len', type=int, default=720, help='Input sequence length')
parser.add_argument('--pred_len', type=int, default=120, help='Prediction length')
parser.add_argument('--batchsize', type=int, default=64, help='Batch size for evaluation')
parser.add_argument('--two_variate', type=bool, default=True, help='Whether to perform two-variate evaluation')
parser.add_argument('--eval_day', type=int, default=209, help='Number of days to evaluate')
args = parser.parse_args()

evaluate(args)