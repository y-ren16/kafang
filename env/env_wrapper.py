import json
import os
from abc import ABC

from env import kafang_stock


class env_with_reward(kafang_stock.KaFangStock):
    def __init__(self, conf, seed=None):
        super(env_with_reward, self).__init__(conf, seed)
        self.conf = conf
        self.seed = seed

    def reset(self):
        """
        整体reset
        :return:
        """
        super(env_with_reward, self).__init__(self.conf, self.seed)

        self.init_info = None
        self.step_cnt = 0
        self.total_r = 0
        self.current_game = 0
        self.total_game = len(self.env_core_list)

        obs, done, info = self.env_core_list[self.current_game].reset()
        self.all_observes = [{"observation": obs, "new_game": True}]
        self.info_his = [info]
        return self.all_observes

    def reset_game(self):
        """
        reset到下一天的数据
        :return:
        """
        self.current_game += 1
        obs, done, info = self.env_core_list[self.current_game].reset()
        self.all_observes = [{"observation": obs, "new_game": True}]
        self.info_his = [info]
        return self.all_observes

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
            obs, done, info = self.env_core_list[self.current_game].step(decoded_order)

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
            obs, done, info = self.env_core_list[self.current_game].reset()  # reset到下一只股票
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
