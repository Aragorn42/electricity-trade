import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.dataset.common import ListDataset
from huggingface_hub import hf_hub_download

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

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
    def __init__(self, MODEL):
        if MODEL == "moirai":
            self.model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
                prediction_length=PDT,
                context_length=CTX,
                patch_size=PSZ,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            )
        elif MODEL == "moirai-moe":
            self.model = MoiraiMoEForecast(
                module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
                prediction_length=PDT,
                context_length=CTX,
                patch_size=16,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            )
        elif MODEL == "moirai2":
            self.model = Moirai2Forecast(
                module=Moirai2Module.from_pretrained(
                    f"Salesforce/moirai-2.0-R-small",
                ),
                prediction_length=120,
                context_length=1680,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
    def forecast(self, pred_len, inputs, args):
        inputs_np = inputs.detach().cpu().numpy()
        ds = ListDataset(
            [
                {
                    "target": row,
                    "start": "2024-01-01 00:00"  # 真实起点
                }
                for row in inputs_np
            ],
            freq="H",  # 按小时采样
        )
        predictor = self.model.create_predictor(batch_size=BSZ)
        forecasts = list(predictor.predict(ds))
        preds = torch.stack([torch.tensor(f[str(args.quant/10.0)]) for f in forecasts], dim=0)
        return preds  # shape [batch, pred_len]