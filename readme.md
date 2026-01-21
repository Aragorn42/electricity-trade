## 训练
对于需要训练的模型, 选择特定比例的数据进行训练, 得到的结果为checkpoint

## 预测
从原始.xlsm读入, 调用预训练checkpoint, 输出给定格式excel

### 运行YingLong
1. 
```bash
pip install xformers transformers

