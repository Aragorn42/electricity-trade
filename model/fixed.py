import torch

class Model:
    def __init__(self, value_workday, value_holiday):
        # 将[24, 1] 的 tensor 展平为 [24]，方便后续在序列维度上进行广播计算
        self.value_workday = torch.tensor(value_workday)
        self.value_holiday = torch.tensor(value_holiday)
        
    def forecast(self, args, history_x, holiday_x, holiday_y):
        """
        params:
            args.pred_len: 预测步长 (int)
            history_x: [Batch, Seq_Len] (数值)
            holiday_x: [Batch, Seq_Len] (0=工作日, 1=假期)
            holiday_y: [Batch, args.pred_len] (未来的假期情况)
        """
        # 1. 构造周期时间索引
        # 因为数据全是24的倍数，预测的第 t 步对应一天中的时刻即为 t % 24
        hours = torch.arange(args.pred_len, device=holiday_y.device) % 24
        
        # 2. 查表：提取每一个预测步长对应的工作日/节假日均值 (Shape: [pred_len])
        pred_w = self.value_workday[hours]
        pred_h = self.value_holiday[hours]
        
        # 3. 组合：利用 holiday_y (0或1) 作为权重进行无分支选择
        # holiday_y shape: [Batch, args.pred_len]
        # pred_w 和 pred_h 会自动广播匹配 Batch 维度
        pred = pred_w * (1 - holiday_y) + pred_h * holiday_y
        
        # 如果下游任务需要最后的特征维度是 [Batch, pred_len, 1]，可取消下行注释
        # pred = pred.unsqueeze(-1) 
        
        return pred