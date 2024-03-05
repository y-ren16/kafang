import argparse
import json
import multiprocessing
from multiprocessing import Pool
import os
import random
from pprint import PrettyPrinter

import numpy as np

from backtest.backtest_oneday import backtest_oneday
from backtest.utils import BacktestMetrics, BacktestStats, ParquetFile
from envs.stock_base_env_cython import StockBaseEnvCython
from mock_market_common.mock_market_data_cython import MockMarketDataCython


def worker(args):
    date, df, code_list, logdir, backtest_mode, TEST_WHITE_CORE_STRATEGY, backtest_datas = args

    backtest_oneday(df, date, code_list, logdir, backtest_mode, TEST_WHITE_CORE_STRATEGY, backtest_datas)


def backtest(logdir, TEST_WHITE_CORE_STRATEGY):
    if TEST_WHITE_CORE_STRATEGY:
        args = None
        model = None
    else:
        pass

    for backtest_mode in ['twoSides']:

        if not os.path.exists(f"{logdir}/backtest_{backtest_mode}"):
            os.makedirs(f"{logdir}/backtest_{backtest_mode}")

        SEED = 1024
        os.environ["PYTHONHASHSEED"] = str(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        # Init Env
        file = ParquetFile()

        # dateList = args.test_date_list
        signal_file_original_rootpath = './data'
        dateList = [name for name in os.listdir(signal_file_original_rootpath) if
                    os.path.isdir(os.path.join(signal_file_original_rootpath, name))]
        dateList.sort()
        print(dateList)
        backtest_datas = multiprocessing.Manager().list()
        processes = []
        envs = []
        tasks = []

        for date in dateList[:]:
            file.filename = "./data/" + date + '/train_data.parquet'
            file.load()
            df = file.data
            code_list = [float(item) for item in df['code'].unique()]
            df = np.array(df)
            # 为每个任务添加一组参数
            tasks.append((date, df, code_list, logdir, backtest_mode, TEST_WHITE_CORE_STRATEGY, backtest_datas))
        max_parallel = 3

        # 使用Pool简化并行执行
        with Pool(max_parallel) as pool:
            pool.map(worker, tasks)

        backtest_metrics = BacktestMetrics(envs, backtest_datas)
        backtest_metrics.make(logdir)
        pp = PrettyPrinter(width=200)
        pp.pprint(backtest_metrics.data)

        backtest_stats = BacktestStats(backtest_metrics.data)
        backtest_stats.make()
        pp.pprint(backtest_stats.data)
        print(backtest_stats.data['daily_pnl_mean_sharped'])


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    print('Start backtesting...')
    TEST_WHITE_CORE_STRATEGY = True
    if TEST_WHITE_CORE_STRATEGY:
        logdir = "./backtest/basic_policy_log_ensemble/"
    else:
        # TODO:Add code to test your reinforcement learning model
        pass
    backtest(logdir, TEST_WHITE_CORE_STRATEGY)
