from env.stock_raw.backtest.utils import ParquetFile
import os
from tqdm import tqdm
import numpy as np

stock_path = "./env/stock_raw"
signal_file_original_rootpath = os.path.join(stock_path, 'data')
dateList = [name for name in os.listdir(signal_file_original_rootpath) if
            os.path.isdir(os.path.join(signal_file_original_rootpath, name))]

print(dateList)

code_list = {}

for date in tqdm(dateList[:]):
    fileList = [name for name in os.listdir(os.path.join(signal_file_original_rootpath, date))]
    for file_name in fileList:
        if len(file_name) != 18:
            code = int(file_name[11:-8])
            if code not in code_list.keys():
                code_list[code] = {'day_count': 0, 'step_count': 0, 'mean_bid_price': [], 'max_bid_price': [],
                                   'min_bid_price': [], 'mean_ask_price': [], 'max_ask_price': [], 'min_ask_price': [],
                                   'delta': []
                                   # 'sign0_score': [], 'sign1_score': [], 'sign2_score': [], 'delta': []
                                   }
            code_list[code]['day_count'] += 1
            file = ParquetFile()
            file.filename = os.path.join(stock_path, "./data/" + date + '/' + file_name)
            file.load()
            df = file.data
            code_list[code]['step_count'] += len(df)
            code_list[code]['mean_bid_price'].append(df['bidPx1'].mean())
            code_list[code]['max_bid_price'].append(df['bidPx1'].max())
            code_list[code]['min_bid_price'].append(df['bidPx1'].min())
            code_list[code]['mean_ask_price'].append(df['askPx1'].mean())
            code_list[code]['max_ask_price'].append(df['askPx1'].max())
            code_list[code]['min_ask_price'].append(df['askPx1'].min())
            x = np.array(0.5 * df['bidPx1'] + 0.5 * df['askPx1'])
            delta = abs(x[:-1] - x[1:]).mean()
            code_list[code]['delta'].append(delta)

with open('stockInfo.csv', 'w') as f:
    f.write(
        'code,day_count,step_count,mean_bid_price,max_bid_price,min_bid_price,mean_ask_price,max_ask_price,min_ask_price,delta\n')
    codes = list(code_list.keys())
    codes.sort()
    for code in codes:
        code_list[code]['mean_bid_price'] = np.mean(code_list[code]['mean_bid_price'])
        code_list[code]['max_bid_price'] = np.mean(code_list[code]['max_bid_price'])
        code_list[code]['min_bid_price'] = np.mean(code_list[code]['min_bid_price'])
        code_list[code]['mean_ask_price'] = np.mean(code_list[code]['mean_ask_price'])
        code_list[code]['max_ask_price'] = np.mean(code_list[code]['max_ask_price'])
        code_list[code]['min_ask_price'] = np.mean(code_list[code]['min_ask_price'])
        code_list[code]['delta'] = np.mean(code_list[code]['delta'])
        f.write(f"{code:d},{code_list[code]['day_count']:d},{code_list[code]['step_count']:d},"
                f"{code_list[code]['mean_bid_price']:.3f},{code_list[code]['max_bid_price']:.3f},"
                f"{code_list[code]['min_bid_price']:.3f},{code_list[code]['mean_ask_price']:.3f},"
                f"{code_list[code]['max_ask_price']:.3f},{code_list[code]['min_ask_price']:.3f},"
                f"{code_list[code]['delta']:.3f}\n")
