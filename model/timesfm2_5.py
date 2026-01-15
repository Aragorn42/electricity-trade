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
        ))

    def forecast(self, pred_len, inputs):
        """
        inputs: [Batch, seq_Len] (Tensor)
        outputs: [Batch, pred_len] (Tensor)
        """
        return self.model.forecast(pred_len, inputs)