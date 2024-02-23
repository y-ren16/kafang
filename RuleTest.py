from env.chooseenv_ori import make as make_ori
from env.chooseenv import make
env_type = "kafang_stock"
env_ori = make_ori(env_type, seed=0)
env = make(env_type, seed=0)

def rule(observation):
    obs = observation['observation']
    
    if obs['signal0'] > 0.8:

        # Long opening
        price = (obs['ap0'] + obs['bp0']) / 2 * (1 + (obs['signal0'] * 0.0001))
        if price < obs['ap0']:
            side = [0, 1, 0]
            volumn = 0
            price = 0
        if obs['ap0'] <= price:
            side = [1, 0, 0]
            volumn = min(obs['av0'], 300 - obs['code_net_position'])
            price = price
    elif obs['signal0'] < -0.8:

        # Short opening
        price = (obs['ap0'] + obs['bp0']) / 2 * (1 + (obs['signal0'] * 0.0001))
        if price > obs['bp0']:
            side = [0, 1, 0]
            volumn = 0
            price = 0
        if obs['bp0'] >= price:
            side = [0, 0, 1]
            volumn = min(obs['bv0'], 300 + obs['code_net_position'])
            price = price
    else:
        side = [0, 1, 0]
        volumn = 0
        price = 0

    return [side, [volumn], [price]]

if  __name__ == "__main__":
    # env.reset()
    # while True:
    #     all_observes = env.all_observes
    #     # all_observes_ori = env_ori.all_observes
    #     action = rule(all_observes[0])
    #     # action_ori = rule(all_observes_ori[0])
    #     all_observes, reward, done, info_before, info_after = env.step([action])
    #     # all_observes_ori, reward_ori, done_ori, info_before_ori, info_after_ori = env_ori.step([action_ori])
    #     # try:
    #     #     assert all_observes == all_observes_ori
    #     # except:
    #     #     print(f"all_observes{all_observes}\n", f"all_observes_ori{all_observes_ori}\n")
    #     # if done_ori == 3:
    #     #     print(7777777)
    #     #     break
    #     if done == 3:
    #         print(8888888)
    #         break
    # # print(all_observes_ori)
    # print(all_observes)
    while True:
        # all_observes = env.all_observes
        all_observes_ori = env_ori.all_observes
        # action = rule(all_observes[0])
        action_ori = rule(all_observes_ori[0])
        # all_observes, reward, done, info_before, info_after = env.step([action])
        all_observes_ori, reward_ori, done_ori, info_before_ori, info_after_ori = env_ori.step([action_ori])
        # try:
        #     assert all_observes == all_observes_ori
        # except:
        #     print(f"all_observes{all_observes}\n", f"all_observes_ori{all_observes_ori}\n")
        if done_ori == 3:
            print(7777777)
            break
        # if done == 3:
        #     print(8888888)
        #     break
    print(all_observes_ori)
    # print(all_observes)