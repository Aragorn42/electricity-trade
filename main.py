from exp.eval import evaluate

import argparse

parser = argparse.ArgumentParser()
# TimesFM-2.5, TimesFM-2.5time, HolidayAvg, NaiveAvg, fixed, DLinear, PatchTST
parser.add_argument('--model_type', type=str, default='Moirai2')
parser.add_argument('--model_path', type=str, default="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/TimesFM2_5_200M")
parser.add_argument('--file_path', type=str, default='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm', help='Path to the Excel file')
parser.add_argument('--seq_len', type=int, default=720, help='Input sequence length')
parser.add_argument('--pred_len', type=int, default=120, help='Prediction length')
parser.add_argument('--batchsize', type=int, default=64, help='Batch size for evaluation')
parser.add_argument('--two_variate', action='store_true')
parser.add_argument('--train_day', type=int, default=438)
parser.add_argument('--val_day', type=int, default=73)
parser.add_argument('--eval_day', type=int, default=209, help='Number of days to evaluate')
parser.add_argument('--need_train', action='store_true')
parser.add_argument('--features', type=str, default="S")
parser.add_argument('--checkpoints_path', type=str, default="./checkpoint")
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--train_epochs', type=int, default=500)
parser.add_argument('--quant', type=int, default=-1)
parser.add_argument('--report', action='store_true')
args = parser.parse_args()
evaluate(args)