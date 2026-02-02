import timesfm

class Model():
    def __init__(self, model_path):
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_path, 
            local_files_only=True
        )
        self.model.compile(timesfm.ForecastConfig(
            max_context=2160,
            max_horizon=256, 
            normalize_inputs=True,
            use_continuous_quantile_head=True, # 概率预测
            force_flip_invariance=True, # 输入-X保证输出-Y
            infer_is_positive=False, # 强制输出非负
            fix_quantile_crossing=True # 分位数保序
        ))

    def forecast(self, pred_len, inputs, args):
        """
        inputs: [Batch, seq_Len] (Tensor)
        outputs: [Batch, pred_len] (Tensor)
        """
        _, y_pred_quant = self.model.forecast(pred_len, inputs)
        return y_pred_quant[:, :, args.quant]