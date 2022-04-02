from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.utils import profile

warnings.simplefilter("error", category=SettingWithCopyWarning)

demo = pd.read_csv('./datasets/train_OwBvO8W/demographics.csv')
print(demo.shape)
event = pd.read_csv('./datasets/train_OwBvO8W/event_calendar.csv')
print(event.shape)
event['YearMonth'] = pd.to_datetime(event['YearMonth'], format='%Y%m')

historical = pd.read_csv('./datasets/train_OwBvO8W/historical_volume.csv')
print(historical.shape)
historical['YearMonth'] = pd.to_datetime(historical['YearMonth'], format='%Y%m')

soda = pd.read_csv('./datasets/train_OwBvO8W/industry_soda_sales.csv')
print(soda.shape)
soda['YearMonth'] = pd.to_datetime(soda['YearMonth'], format='%Y%m')
industry = pd.read_csv('./datasets/train_OwBvO8W/industry_volume.csv')
print(industry.shape)
industry['YearMonth'] = pd.to_datetime(industry['YearMonth'], format='%Y%m')

price = pd.read_csv('./datasets/train_OwBvO8W/price_sales_promotion.csv')
print(price.shape)
price['YearMonth'] = pd.to_datetime(price['YearMonth'], format='%Y%m')
weather = pd.read_csv('./datasets/train_OwBvO8W/weather.csv')
print(weather.shape)
weather['YearMonth'] = pd.to_datetime(weather['YearMonth'], format='%Y%m')

sku = historical.merge(price, on=['Agency', 'SKU', 'YearMonth'], how='left')
sku = sku.merge(soda, on=['YearMonth'], how='left')
sku = sku.merge(industry, on='YearMonth', how='left')
sku = sku.merge(event, on=['YearMonth'], how='left')
print(sku.shape)

agency = weather.merge(demo, on=['Agency'], how='left')
print(agency.shape)

df = sku.merge(agency, on=['YearMonth', 'Agency'], how='left')
print(df.shape)

# add time index
df["time_idx"] = df["YearMonth"].dt.year * 12 + df["YearMonth"].dt.month
df["time_idx"] -= df["time_idx"].min()

# add additional features
df["month"] = df["YearMonth"].dt.month.astype(str).astype("category")  # categories have be strings
df["log_volume"] = np.log(df.Volume + 1e-8)
df["avg_volume_by_sku"] = df.groupby(["time_idx", "SKU"], observed=True).Volume.transform("mean")
df["avg_volume_by_agency"] = df.groupby(["time_idx", "Agency"], observed=True).Volume.transform("mean")
df["id"] = np.arange(df.shape[0])
df["timeseries"] = df["id"].apply(lambda x: x % 350)
df["discount_in_price"] = 100 * (df["Promotions"])/df["Price"]
df["discount_in_price"] = df["discount_in_price"].fillna(0)
df = df.drop(["id"], axis=1)
aa = df.describe()
print("data precess end")
