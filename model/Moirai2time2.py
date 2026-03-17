import torch
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.dataset.common import ListDataset
from huggingface_hub import hf_hub_download

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

import torch
import numpy as np
import math
from gluonts.dataset.common import ListDataset
from einops import rearrange
MODEL = "moirai2"  # model name: choose from {'moirai', 'moirai-moe', 'moirai2'}
SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 120  # prediction length: any positive integer
CTX = 200  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer

# Read data into pandas DataFrame
url = (
    "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
    "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
)
df = pd.read_csv(url, index_col=0, parse_dates=True)

# Convert into GluonTS dataset
ds = PandasDataset(dict(df))

# Split into train/test set
train, test_template = split(
    ds, offset=-TEST
)  # assign last TEST time steps as test set

# Construct rolling window evaluation
test_data = test_template.generate_instances(
    prediction_length=PDT,  # number of time steps for each prediction
    windows=TEST // PDT,  # number of windows in rolling window evaluation
    distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
)
class Model():
    def __init__(self):
        # if MODEL == "moirai":
        #     self.model = MoiraiForecast(
        #         module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
        #         prediction_length=PDT,
        #         context_length=CTX,
        #         patch_size=PSZ,
        #         num_samples=100,
        #         target_dim=1,
        #         feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        #         past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        #     )
        # elif MODEL == "moirai-moe":
        #     self.model = MoiraiMoEForecast(
        #         module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
        #         prediction_length=PDT,
        #         context_length=CTX,
        #         patch_size=16,
        #         num_samples=100,
        #         target_dim=1,
        #         feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        #         past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        #     )
        # elif MODEL == "moirai2":
        self.model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(
                f"Salesforce/moirai-2.0-R-small",
            ),
            prediction_length=120,
            context_length=1680,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(torch.device("cuda"))

    def forecast(self, pred_len, inputs, args, quants=None):
        """
        策略：
        1. 将输入 reshape 为 [Batch, Days, 24]。
        2. 循环 24 次（0-23时），每次取出一个 slice [Batch, Days, 1] 作为独立的日频序列。
        3. 对每个小时的序列预测 pred_days (即 pred_len / 24) 步。
        4. 对每个小时应用特定的分位数。
        5. 拼接结果并还原时间顺序。
        """
        B, seq_len = inputs.shape
        period = 24
        num_days = seq_len // period
        pred_days = math.ceil(pred_len / period)
        # [Batch, seq_len] -> [Batch, num_days, 24]
        inputs_np = inputs.detach().cpu().numpy()

        x_reshaped = inputs_np[:, :num_days * period].reshape(B, num_days, period)

        hourly_results = []
        for h in range(period):
            input_h = x_reshaped[:, :, h]  # [B, num_days]
            preds_h = self.model.predict(
                past_target=[row for row in input_h]
            )  # [B, Q, future_time]
            hourly_results.append(preds_h[:, :, :pred_days])  # [B, Q, pred_days]

        # [Batch, 24, pred_days, Num_Quantiles]
        forecast_stacked = np.stack(hourly_results, axis=1)
        return rearrange(forecast_stacked, 'b h d q -> b (d h) q')