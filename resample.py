from env.stock_raw.backtest.utils import ParquetFile
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

def vectorized_resample(df, sampling_interval, on='serverTime'):
    # 创建采样时间点
    start_time = 93000000.0
    end_time = df[on].iloc[-1]
    sample_times = np.arange(start_time, end_time, sampling_interval)
    
    # 找到每个采样时间点最接近的索引
    indices = np.searchsorted(df[on], sample_times, side="left")
    
    # 确保索引在DataFrame范围内
    # indices = np.clip(indices, 0, len(df) - 1)
    
    # 选取最接近采样时间点的行
    resampled_df = df.iloc[indices].copy()
    resampled_df.reset_index(drop=True, inplace=True)
    
    return resampled_df

def df_resample(df, sampling_interval, on='serverTime'):
    # 生成采样点时间戳
    start_time = 93000000.0
    end_time = df[on].iloc[-1]
    sample_times = np.arange(start_time, end_time, sampling_interval)
    # print(len(sample_times))
    
    # 准备采样结果容器
    sampled_indices = []
    current_sample_index = 0
    cache = None
    cache_index = 0

    # 对于DataFrame中的每一行
    for index, row in df.iterrows():
        if current_sample_index >= len(sample_times):
            break
        # 当前行的时间戳
        current_time = row[on]
        # 缓存上一行内容
        if cache is None:
            cache = row
            cache_index = index
        if current_time >= sample_times[current_sample_index]:
            diff_pre = sample_times[current_sample_index] - cache[on]
            diff_next = current_time - sample_times[current_sample_index]
            if diff_next < diff_pre:
                sampled_indices.append(index)
            else:
                sampled_indices.append(cache_index)
            current_sample_index += 1
        cache = row
        cache_index = index

    # 使用找到的索引从原始DataFrame中选择行
    resampled_df = df.loc[sampled_indices].copy()

    # 重置索引
    resampled_df.reset_index(drop=True, inplace=True)
    
    return resampled_df

if __name__ == "__main__":
    stock_path = "./env/stock_raw"
    signal_file_original_rootpath = os.path.join(stock_path, 'data')
    dateList = [name for name in os.listdir(signal_file_original_rootpath) if
                        os.path.isdir(os.path.join(signal_file_original_rootpath, name))]
    print(dateList)
    os.makedirs(os.path.join(stock_path, "./data_resampled"), exist_ok=True)
    for date in tqdm(dateList[:]):
        print(date)
        if date == '20200225' or date == '20200224' or date == '20200221' or date == '20200220':
            continue
        file = ParquetFile()
        os.makedirs(os.path.join(stock_path, "./data_resampled/" + date), exist_ok=True)
        file.filename = os.path.join(stock_path, "./data/" + date + '/train_data.parquet')
        file.load()
        df_ori = file.data
        code_list = df_ori['code'].unique()
        print(len(code_list))
        # assert len(code_list) == 496
        # assert len(code_list) == 100
        code_nums_each = []
        for code in tqdm(code_list):
            df = df_ori[df_ori['code'] == code]
            code_nums_each.append(len(df))
            # start_time = 93000000.0
            # end_time = 145900000.0
            assert df.iloc[0]["serverTime"] > 93000000.0
            # assert df.iloc[-1]["serverTime"] < 145900000.0

            # resampled df ever 5 seconds
            # df_resampled = df_resample(df, sampling_interval=5000.0, on='serverTime')
            df_resampled = vectorized_resample(df, sampling_interval=5000.0, on='serverTime')

            new_file = ParquetFile()
            new_file.filename = os.path.join(stock_path, "./data_resampled/" + date + '/train_data_' + str(int(code)) + '.parquet')
            new_file.data = df_resampled
            new_file.dump()
            continue
        assert sum(code_nums_each) == len(df_ori)
        continue