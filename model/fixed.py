class Model:
    def __init__(self, value):
        # [24, 1] tensor
        self.value = value
    def forecast(self, horizon, inputs):
        """
        inputs: [Batch, Seq_Len]
        return: [Batch, horizon]
        """
        return self.value.repeat(inputs.shape[0], horizon)