from envs.utils import Order
from backtest.SACAdjRule_3_0.submission import my_controller


def base_taker_policy(obs, info):
    x1 = 0.8
    if obs['signal0'] > x1:

        # Long opening
        price = (obs['ap0'] + obs['bp0']) / 2 * (1 + (obs['signal0'] * 0.0001))
        if price < obs['ap0']:
            order = Order(side=1, price=0, volume=0)
        if obs['ap0'] <= price:
            order = Order(side=0, price=price, volume=min(obs['av0'], 300 - info['code_net_position']))

    elif obs['signal0'] < -x1:

        # Short opening
        price = (obs['ap0'] + obs['bp0']) / 2 * (1 + (obs['signal0'] * 0.0001))
        if price > obs['bp0']:
            order = Order(side=1, price=0, volume=0)
        if obs['bp0'] >= price:
            order = Order(side=2, price=price, volume=min(obs['bv0'], 300 + info['code_net_position']))

    else:
        order = Order(side=1, price=0, volume=0)

    return order

def rl_policy(obs, info):
    all_observes = {}
    all_observes['observation'] = obs
    side, volume, price = my_controller(all_observes, None, False)
    # one hot to int
    side = side.index(1)
    if isinstance(volume, list):
        volume = volume[0]
    if isinstance(price, list):
        price = price[0]
    order = Order(side=side, price=price, volume=volume)
    return order