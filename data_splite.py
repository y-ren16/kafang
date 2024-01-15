from env.stock_raw.backtest.utils import ParquetFile
import os
from tqdm import tqdm
stock_path = "./env/stock_raw"
signal_file_original_rootpath = os.path.join(stock_path, 'data')
dateList = [name for name in os.listdir(signal_file_original_rootpath) if
                    os.path.isdir(os.path.join(signal_file_original_rootpath, name))]
print(dateList)
for date in tqdm(dateList[:]):
    file = ParquetFile()
    file.filename = os.path.join(stock_path, "./data/" + date + '/train_data.parquet')
    file.load()
    df_ori = file.data
    code_list = df_ori['code'].unique()
    print(len(code_list))
    code_nums_each = []
    for code in code_list:
        df = df_ori[df_ori['code'] == code]
        code_nums_each.append(len(df))
        new_file = ParquetFile()
        new_file.filename = os.path.join(stock_path, "./data/" + date + '/train_data_' + str(code) + '.parquet')
        new_file.data = df
        new_file.dump()
    assert sum(code_nums_each) == len(df_ori)