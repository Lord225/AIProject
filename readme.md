![image](https://github.com/Lord225/reinforcment-learning-diag-chess/assets/49908210/256b30a6-1d68-4198-9231-41dd030135f5)# install
```
pip install -r .\requirements.txt
```

# run tensorboard
```
run_tb.bat
sh run_tb.sh
```

```
ctrl+shift+p >Python:Launch TensorBoard
```

# run tests
* Install Testing extension for VSCode
* Click run all
* observe

# results
![image](https://github.com/Lord225/reinforcment-learning-diag-chess/assets/49908210/83437404-c18c-4b90-9b77-6cc30a044dff)
![image](https://github.com/Lord225/reinforcment-learning-diag-chess/assets/49908210/e99f5680-8f21-4abb-8c48-acf43c7a37f4)

![image](https://github.com/Lord225/reinforcment-learning-diag-chess/assets/49908210/2bf45f9f-b49f-4ad5-b0f7-27d34f1410d1)
![image](https://github.com/Lord225/reinforcment-learning-diag-chess/assets/49908210/773576e9-8d8a-486f-a79f-11c8a2247efc)
# Example game
![image](https://github.com/Lord225/reinforcment-learning-diag-chess/assets/49908210/b628ab04-b2e1-4aae-a10b-0b920524c090)
![image](https://github.com/Lord225/reinforcment-learning-diag-chess/assets/49908210/d6bdf5b7-8929-4fd2-84e8-39dcb524f9da)
![image](https://github.com/Lord225/reinforcment-learning-diag-chess/assets/49908210/9e4a1b9f-9f4d-4f5d-82ae-df76265ec485)
![image](https://github.com/Lord225/reinforcment-learning-diag-chess/assets/49908210/ef218635-0c45-4c73-99bf-c551a2d019a4)
![image](https://github.com/Lord225/reinforcment-learning-diag-chess/assets/49908210/34a5e4c4-98f7-4825-9df9-77216cf6a1b3)


# reproduce
> `python ./src/chess_lr_6.py`
Make sure you have right parameters in `chess_lr_6.py`
```py
batch_size = 2048
discount_rate = 0.1 # 0.5, 0.9
episodes = 1000000
minibatch_size = 128 # 256
train_iters_per_episode = 16 # batch_size/minibatch_size 
train_interval = 3 # 1
max_steps_per_episode = 15 
target_update_freq = 500
replay_memory_size = 10_000 # 15_000
save_freq = 250 # 500
eps_decay_len = 1000 # 100
eps_min = 0.05
lr = 1e-3 # 3e-4 1e-4 3e-5 1e-5
```
You can now resume training with new  hiperparameters (mainly minibatch_size, eps_decay_len=100 & lr)
> `python ./src/chess_lr_6.py --resume /models/name_of_last_checkpoint.h5`
You can run algorithm for around 15-30k iterations.

# documentation
https://docs.google.com/document/d/1mwrjl8Hykerf2FBW2TFN4IuPMoahdylG-ok0QbePG2g/edit?usp=sharing
