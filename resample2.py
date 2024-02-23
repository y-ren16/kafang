from env.stock_raw.backtest.utils import ParquetFile
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

stock_path = "./env/stock_raw"

signal_file_original_rootpath = os.path.join(stock_path, 'data1')

os.makedirs(os.path.join(stock_path, "./data_resampled1"), exist_ok=True)

dateList = [name for name in os.listdir(signal_file_original_rootpath) if
                        os.path.isdir(os.path.join(signal_file_original_rootpath, name))]

for date in tqdm(dateList[:]):
    file = ParquetFile()
    file.filename = os.path.join(signal_file_original_rootpath, date, 'train_data.parquet')
    file.load()
    df_ori = file.data

    resampled_index = []
    init_time = pd.to_datetime('93000000', format='%H%M%S%f')  # 九点半开始
    last_time = init_time
    eventTime = pd.to_datetime(df_ori['eventTime'], format='%H%M%S%f')
    for i in range(eventTime.shape[0]):
        if i == 0 or df_ori['code'][i] != df_ori['code'][i - 1]:
            assert -5 <= (eventTime[i] - init_time).total_seconds() <= 5, (df_ori['code'][i], eventTime[i], i)
            resampled_index.append(i)
            last_time = init_time + pd.Timedelta(seconds=5)
        elif eventTime[i] >= last_time:
            if (eventTime[i] - last_time).total_seconds() < (last_time - eventTime[i - 1]).total_seconds():
                resampled_index.append(i)
            else:
                resampled_index.append(i - 1)
            last_time += pd.Timedelta(seconds=5)
    df_resample = df_ori.iloc[resampled_index]
    df_resample = df_resample.reset_index(drop=True)


    os.makedirs(os.path.join(stock_path, "data_resampled1", date), exist_ok=True)
    new_file = ParquetFile()
    new_file.filename = os.path.join(stock_path, "data_resampled1", date, 'train_data.parquet')
    new_file.data = df_resample
    new_file.dump()
