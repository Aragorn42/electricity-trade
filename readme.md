本项目是一个用于电力交易价格预测的平台. 项目集成了统计学基准模型, 专用时序深度学习模型和序基础大模型等多种方法, 对比不同技术路线在电力交易场景下的预测性能. 并选取了其中效果最好的几个模型进行了综合决策. 

## 项目简介

随着电力市场的波动性增加, 准确的价格预测对于制定交易策略至关重要. 本项目涵盖了以下三个维度的预测模型：

1.  **统计基准**：用于确立预测性能的下限. 
2.  **深度学习模型**：针对时间序列预测的专用小模型. 
3.  **时序基础大模型**：基于大规模预训练数据的零样本预测模型. 

## 模型列表

### 1. 基准模型
*   **NaiveAvg**: 直接平均模型, 使用历史窗口的均值作为预测值. 
*   **HolidayAvg**: 节假日平均模型, 根据日期类型（工作日/节假日）分别计算历史均值. 

### 2. 深度学习模型
*   **DLinear**: 基于线性分解架构的模型, 结构简单但在长序列预测中表现稳健. 
*   **PatchTST**: 基于 Patch 技术和 Transformer 骨干网络的模型, 能够有效捕捉局部语义和长期依赖. 

### 3. 时序基础大模型
集成了当前业内最先进的大规模预训练模型：
*   **TimesFM 2.5** (Google)
*   **Chronos 2** (Amazon)
*   **Moirai 2** (Salesforce)
*   **YingLong** (Alibaba)
*   **FalconTST** (Alibaba)
*   **Timer**
*   **Sundial**
*   **TimeMOE**

带有后缀"time"的模型表示将输入按小时拆分后分别传入模型进行预测和输出, 然后拼接成整个输出;

后缀"time2"代表在将输入分别传入模型的基础上对输出值使用不同的分位数.
## 目录结构说明

```text
electricity-trade/
├── dataset/            # 数据存放目录
├── exp/                # 实验执行模块 (包含训练 Train 和评估 Eval)
├── model/              # 模型定义与统一接口封装
├── script/             # 自动化测试脚本 (Bash 脚本)
├── utils/              # 工具库
└──  main.py            # 程序主入口
```
## 使用说明
项目统一通过 main.py 进行调用, 主要包含训练和评估两个阶段.
1. 运行深度学习模型 (训练 + 评估)
对于需要从头训练的模型（如 DLinear, PatchTST）， 加入参数'need_train'：

```Bash
for model in 'DLinear' 'PatchTST';do
    python -u main.py \
    --model_type=$model \
    --seq_len=336 \
    --pred_len=120 \
    --batchsize=64 \
    --need_train \
    --report \
    --eval_day=209
done
```
2. 仅评估
其他模型都可以直接进行评估, 对于时序大语言模型会自动下载checkpoint.
```Bash
for j in 720;do
    for model in 'HolidayAvg';do
        python -u main.py \
        --model_type=$model \
        --seq_len=$j \
        --pred_len=120 \
        --batchsize=64 \
        --report \
        --eval_day=209
    done
done
```
3. 如果需要对多个模型的结果进行综合决策, 需要运行analys.ipynb, 该文件会读取"output_report.xlsx"当中模型的结果做出综合决策并给出估计的准确率.
4. 其他参数说明

- --report 加上这个参数模型会将输出预测结果和是否准确保存为一个excel文件.
- --two_variate 加上这个参数模型会分别预测日前价格和实时价格, 用其差值作为对差价的预测值.
- --quant 对于支持使用分位数预测的模型如Moirai, Chronos-2, TimesFM-2.5, YingLong, 这个参数可以传入不同分位数来达到"保守"/ "激进"的预测风格