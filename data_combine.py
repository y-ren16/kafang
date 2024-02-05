from env.stock_raw.backtest.utils import ParquetFile  # 假设这是有效的导入路径
import os
import pandas as pd
from tqdm import tqdm

stock_path = "./env/stock_raw"
signal_file_original_rootpath = os.path.join(stock_path, 'data_resampled')
dateList = [name for name in os.listdir(signal_file_original_rootpath) if
            os.path.isdir(os.path.join(signal_file_original_rootpath, name))]

print(dateList)

for date in tqdm(dateList):
    print(date)
    # 跳过特定日期
    if date in {'20200225', '20200224', '20200221', '20200220'}:
        continue
    
    all_dfs = []  # 用于存储当天所有股票数据的列表
    for file_code in os.listdir(os.path.join(stock_path, "data_resampled", date)):
        if file_code.endswith(".parquet"):
            code = file_code.split(".")[0][11:]
            # print(code)
            # continue
            file_path = os.path.join(stock_path, "data_resampled", date, file_code)
            df = pd.read_parquet(file_path)  # 直接加载Parquet文件为DataFrame
            all_dfs.append(df)
    
    if all_dfs:
        # 合并当天所有股票数据的DataFrame
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # 按['serverTime']列排序
        # combined_df_sorted = combined_df.sort_values(by=['serverTime'])
        # 可以选择保存合并后的DataFrame为新的Parquet文件
        combined_df.to_parquet(os.path.join(stock_path, "data_resampled", date, "train_data.parquet"))