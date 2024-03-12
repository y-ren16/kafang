 # 玛卡巴卡队源码复现说明

在官方提供的源码基础上，对环境进行微调（增加reward、修改reset函数等），并使用降采样至五秒后的数据训练Soft Actor-Critic。

## 环境配置

在官方所用的环境配置的基础上，需要额外安装pytorch用于神经网络训练，安装tqdm和tensorboard。环境配置步骤如下：

```bash
conda create -n stock-venv python==3.7.5  
conda activate stock-venv
pip install -r requirements.txt
```

## 数据预处理
如果是原始数据，将其放到文件夹./env/stock_raw/data.ori，使用data_preprocess.py对其按照eventTime进行5秒的降采样
如果是降采样后的数据，将其放到文件夹./env/stock_raw/data
```bash
python data_preprocess.py
```

## 模型训练

```bash
CUDA_VISIBLE_DEVICES=0 python RL_train/SACAdjRule2.py \
    --save-dir ./output_SAC_rule2/test

```

- 通过上述命令进行强化学习模型训练
- 模型文件保存至`./output_SAC_rule2/test/models`文件夹
- 训练过程损失函数保存至`./output_SAC_rule2/test/log`文件夹，可使用tensorboard查看`tensorboard --logdir=./output_SAC_rule2 --port=6006`。


## 测试

- 训练完成后，将模型文件`./output_SAC_rule2/test/models/RL_part_1000k.pt`复制至`./env/stock_raw/backtest/sac_rule2/`文件夹，该文件夹即为比赛提交文件夹。
- 测试时，修改`env/stock_raw/test_new.py`的第84行作为保存多线程测试结果的目录，默认是`./backtest/sac_policy_log/`
