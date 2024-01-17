import json
import os
from abc import ABC
from env import kafang_stock
import sys
from pathlib import Path
from .stock_raw.backtest.utils import ParquetFile
from .stock_raw.mock_market_common.mock_market_data_cython import MockMarketDataCython
from .stock_raw.envs.stock_base_env_cython import StockBaseEnvCython
import numpy as np

CURRENT_PATH = str(Path(__file__).resolve().parent.parent)
taxing_path = os.path.join(CURRENT_PATH)
sys.path.append(taxing_path)
print(CURRENT_PATH)
stock_path = os.path.join(CURRENT_PATH, 'env/stock_raw')
sys.path.append(stock_path)


class env_with_reward(kafang_stock.KaFangStock):
    def __init__(self, conf, seed=None, dataList=None):
        super(env_with_reward, self).__init__(conf, seed, dataList)
        self.conf = conf
        self.seed = seed
        self.info_his = []

    def reset_env_core(self):
        """
        将初始化env_core，env_core的数据为第self.dateList[self.current_game]天的数据，与基类KaFangStock的reset_env_core相比，增加了保存初始info
        :return:
        """
        date = self.dateList[self.current_game]
        file = ParquetFile()
        file.filename = os.path.join(stock_path, "./data/" + date + '/train_data.parquet')
        file.load()
        df = file.data
        code_list = []
        for item in df['code'].unique():
            code_list.append(float(item))
        df = np.array(df)
        mock_market_data = MockMarketDataCython(df)
        self.env_core = StockBaseEnvCython(date, code_list, mock_market_data, limit_of_netpos=300)

        obs, done, info = self.env_core.reset()
        self.all_observes = [{"observation": obs, "new_game": True}]
        self.info_his = [info]
        return

    def step(self, action):
        """
        Action format:
        [side, volume, price]
        """
        self.is_valid_action(action)
        action_array = action[0]
        decoded_order = self.convert_action(action_array)

        # side = 2  ##random.choice([0,1,2])
        # volume = min(1, self.all_observes[0]['bv0'])#random.randrange(0,2)
        # price = self.all_observes[0]['bp0']-0.1

        try:
            obs, done, info = self.env_core.step(decoded_order)

        except ValueError as v:
            print(f'Current game terminate early due to error {v}')
            done = True
            obs = {}
            info = None

        # self.all_observes = [obs]
        self.step_cnt += 1
        self.info_his.append(info)
        # reward = info['code_pnl'] - self.info_his['code_pnl']  # 以这步收益为回报
        reward = self.compute_reward()

        if done == 2:
            obs, done, info = self.env_core.reset()  # reset到下一只股票
            self.info_his = [info]
            self.all_observes = [{"observation": obs, "new_game": False}]
        elif done and (self.current_game<self.total_game-1):
            obs = self.reset_game()
            self.all_observes = obs
        elif done and (self.current_game==self.total_game-1):
            self.done = True
            self.all_observes = [{"observation": obs, "new_game": False}]
        else:
            self.all_observes = [{"observation": obs, "new_game": False}]


        if self.done:
            self._load_backtest_data()
            self.compute_final_stats()
            self.set_n_return()

        return self.all_observes, reward, done>0, self.info_his, ''

    def compute_reward(self):
        pass


class env_with_pnl_reward(env_with_reward, ABC):
    def __init__(self, conf, seed=None):
        super(env_with_pnl_reward, self).__init__(conf, seed)

    def compute_reward(self):
        if len(self.info_his) > 1:
            return (self.info_his[-1]['code_pnl'] - self.info_his[-2]['code_pnl']) / self.all_observes[0]['observation']['ap0_t0']
        else:
            return 0
