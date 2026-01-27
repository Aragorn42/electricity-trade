export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com 
current_time=$(date +'%H:%M:%S')
# best for chronos-2time : input_len 120*24=2880, quant 12 single, 65.09%
# chronos: input len 1440 quant 12 single, 65.03%
# chronos holiday
# timesfm input 720 quant 6(in 10) single, 65.15%
# timesfm time input len 1080 quant6 65.29%
# YingLong input len 24*32 quant 80(in 100) 64.96%
echo $current_time
# for j in 720;do
#     for model in 'NaiveAvg' 'HolidayAvg';do
#         python -u main.py \
#         --model_type=$model \
#         --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/TimesFM2_5_200M" \
#         --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#         --seq_len=$j \
#         --pred_len=120 \
#         --batchsize=64 \
#         --report \
#         --eval_day=209
#     done
# done
# for j in 720;do
#     for i in 10;do
#         python -u main.py \
#         --model_type "Chronos-2holiday" \
#         --seq_len $j \
#         --pred_len 120 \
#         --quant $i \
#         --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Chronos2"
#     done
# done
# for j in 2880;do
#     for i in 10;do
#         python -u main.py \
#         --model_type "Chronos-2time2" \
#         --seq_len $j \
#         --pred_len 120 \
#         --quant $i \
#         --report \
#         --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Chronos2"
#     done
# done


# for j in 1440;do
#     for i in 10;do
#         python -u main.py \
#         --model_type "Chronos-2time" \
#         --seq_len $j \
#         --pred_len 120 \
#         --quant $i \
#         --report \
#         --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Chronos2"
#     done
# done
# for j in 1440;do
#     for i in 10;do
#         python -u main.py \
#         --model_type "Chronos-2holiday" \
#         --seq_len $j \
#         --pred_len 120 \
#         --two_variate=True \
#         --quant $i \
#         --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Chronos2"
#     done
# done
# for model in 'DLinear' 'PatchTST';do
#     python -u main.py \
#     --model_type=$model \
#     --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/TimesFM2_5_200M" \
#     --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#     --seq_len=336 \
#     --pred_len=120 \
#     --batchsize=64 \
#     --need_train \
#     --report \
#     --eval_day=209
# done
# for j in 720;do
#     for i in 5;do
#         for model in 'TimesFM-2.5';do
#             python -u main.py \
#             --model_type=$model \
#             --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/TimesFM2_5_200M" \
#             --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#             --seq_len=$j \
#             --pred_len=120 \
#             --batchsize=64 \
#             --eval_day=209 \
#             --report \
#             --quant=$i
#         done
#     done
# done
# for j in 1080;do
#     for i in 5;do
#         for model in 'TimesFM-2.5time';do
#             python -u main.py \
#             --model_type=$model \
#             --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/TimesFM2_5_200M" \
#             --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#             --seq_len=$j \
#             --pred_len=120 \
#             --batchsize=64 \
#             --eval_day=209 \
#             --quant=$i \
#             --report
#         done
#     done
# done
# for j in 768;do
#     for i in 60;do
#         for model in 'YingLong';do
#             python -u main.py \
#             --model_type=$model \
#             --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#             --seq_len=$j \
#             --pred_len=120 \
#             --batchsize=32 \
#             --eval_day=209 \
#             --quant=$i \
#             --report
#         done
#     done
# done
# for i in 720;do
#     for model in 'TimeMoE';do
#         /home/liym/miniconda3/envs/tfm/bin/python -u main.py \
#         --model_type=$model \
#         --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#         --seq_len=$i \
#         --pred_len=120 \
#         --batchsize=64 \
#         --report \
#         --eval_day=209
#     done
# done
# for i in 480;do
#     for model in 'Timer';do
#         /home/liym/miniconda3/envs/tfm/bin/python -u main.py \
#         --model_type=$model \
#         --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#         --seq_len=$i \
#         --pred_len=120 \
#         --batchsize=64 \
#         --report \
#         --eval_day=209
#     done
# done
for i in 480;do
    for model in 'FalconTST';do
        python main.py \
        --model_type=$model \
        --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
        --seq_len=$i \
        --pred_len=120 \
        --report \
        --batchsize=64 \
        --eval_day=209
    done
done
current_time=$(date +'%H:%M:%S')
echo $current_time
