from abc import ABC
from env.kafang_stock import *
import numpy as np


class env_with_reward(KaFangStock):
    def __init__(self, conf, seed=None, dateList=None):
        super(env_with_reward, self).__init__(conf, seed, dateList)
        self.conf = conf
        self.seed = seed
        self.info_his = []

        self.last_pnl = None

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
            # print(1)
            # self.last_pnl = (obs['code'], obs['code_pnl'])
            # print(obs)
            obs, _, info = self.env_core.reset()  # reset到下一只股票
            self.info_his = [info]
            self.all_observes = [{"observation": obs, "new_game": False}]
        elif done and (self.current_game<self.total_game-1):
            # print(2)
            self.last_pnl = (obs['code'], obs['code_pnl'])
            obs = self.reset_game()
            self.all_observes = obs
        elif done and (self.current_game==self.total_game-1):
            # print(3)
            self.last_pnl = (obs['code'], obs['code_pnl'])
            self.done = True
            self.all_observes = [{"observation": obs, "new_game": False}]
        else:
            self.all_observes = [{"observation": obs, "new_game": False}]


        if self.done:
            self._load_backtest_data()
            self.compute_final_stats()
            self.set_n_return()

        return self.all_observes, reward, done>0, self.info_his[-1], ''

    def compute_reward(self):
        pass


class env_with_pnl_reward(env_with_reward, ABC):
    def __init__(self, conf, seed=None, dateList=None):
        super(env_with_pnl_reward, self).__init__(conf, seed, dateList)

    def compute_reward(self):
        # code_pnl的计算方式如下：code_positional_pnl+code_cash_pnl-code_handling_fee
        # 其中code_positional_pnl=code_net_position*10*(ap0+bp0)/2，即使用最高买价和最低卖价的均值作为股票估价来计算持仓估值
        # code_cash_pnl为现金变化量（不考虑手续费）
        # code_handling_fee为手续费，买/卖双方均需支付，比例为0.00007

        if len(self.info_his) > 1:
            return (self.info_his[-1]['code_pnl'] - self.info_his[-2]['code_pnl']) / self.all_observes[0]['observation']['ap0_t0']
        else:
            return 0
