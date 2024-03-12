 

# 玛卡巴卡队源码复现

在官方提供的源码基础上，对环境进行微调（增加reward、修改reset函数等），并使用降采样至五秒后的数据训练Soft Actor-Critic。本源码中不包含所用数据，需要手动复制到.\env\stock_raw\data目录下。

## 环境配置

在官方所用的环境配置的基础上，需要额外安装pytorch用于神经网络训练。环境配置步骤如下：

```bash
conda create -n stock-venv python==3.7.5  
conda activate stock-venv
pip install -r requirements.txt
```

训练命令如下：

```bash
CUDA_VISIBLE_DEVICES=0 python RL_train/SACAdjRule2.py --save-dir ./output_SAC_rule2/test

```

通过上述命令，可以训练过程文件会保存至./output_SAC_rule2/test/models文件夹，训练过程损失函数保存至./output_SAC_rule2/test/log文件夹。

## 测试

训练完成后，可以将./output_SAC_rule2/test/models/RL_part_1000k.pt复制至./env/stock_raw/backtest/sac_rule2文件夹，该文件夹即可作为提交文件。
