export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_ENDPOINT=https://hf-mirror.com
export DISABLE_VERSION_CHECK=1
current_time=$(date +'%H:%M:%S')
echo $current_time
# best for chronos-2time : input_len 120*24=2880, quant 12 single, 65.09%
# chronos: input len 1440 quant 12 single, 65.03%
# chronos holiday
# timesfm input 720 quant 6(in 10) single, 65.15%
# timesfm time input len 1080 quant6 65.29%
# YingLong input len 24*32 quant 80(in 100) 64.96%

for j in 720;do
    for i in 10;do
        python -u main.py \
        --model_type "qwen35b" \
        --seq_len $j \
        --pred_len 120 \
        --eval_day=209 \
        --report \
        --batchsize=2 \
        --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Chronos2"
    done
done

# for j in 720;do
#     for model in 'HolidayAvg';do
#         python -u main.py \
#         --model_type=$model \ 
#         --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#         --seq_len=$j \
#         --pred_len=120 \
#         --batchsize=64 \
#         --report \
#         --train_day=100 \
#         --eval_day=268
#     done
# done
# for j in 0;do
#     for model in 'fixed';do
#         python -u main.py \
#         --model_type=$model \
#         --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#         --seq_len=$j \
#         --pred_len=120 \
#         --batchsize=64 \
#         --report \
#         --train_day=100 \
#         --eval_day=365
#     done
# done
# for i in 720;do
#     for j in -1;do
#         for model in 'moirai2time2';do
#             python3 -u main.py \
#             --model_type=$model \
#             --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#             --seq_len=$i \
#             --pred_len=120 \
#             --batchsize=64 \
#             --quant=$j \
#             --report \
#             --eval_day=268
#         done
#     done
# done

# for j in 768;do
#     for i in 80;do
#         for model in 'YingLongtime';do
#             python -u main.py \
#             --model_type=$model \
#             --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#             --seq_len=$j \
#             --pred_len=120 \
#             --batchsize=32 \
#             --eval_day=268 \
#             --quant=$i \
#             --report
#         done
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
#         --eval_day=268 \
#         --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/Chronos2"
#     done
# done

# for j in 1080;do
#     for i in 6;do
#         for model in 'TimesFM-2.5time';do
#             python -u main.py \
#             --model_type=$model \
#             --model_path="/home/liym/code/ElectricityTrade/electricity-trade/checkpoint/TimesFM2_5_200M" \
#             --file_path='/home/liym/code/ElectricityTrade/electricity-trade/dataset/日前实时套利计算.xlsm' \
#             --seq_len=$j \
#             --pred_len=120 \
#             --batchsize=64 \
#             --eval_day=268 \
#             --quant=$i \
#             --report
#         done
#     done
# done 

# current_time=$(date +'%H:%M:%S')
# echo $current_time
