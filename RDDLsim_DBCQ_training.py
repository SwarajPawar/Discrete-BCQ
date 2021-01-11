#!/usr/bin/env python
# coding: utf-8




from main import train_DBCQ, generate_buffer_from_dataset, dargs


import pandas as pd

dataset = "crossing_traffic"
samples = 500000
steps = 10
lookback = 3
variables = 5
data = pd.read_csv(f"{dataset}_{samples}x{steps}x{lookback}.tsv", sep='\t', index_col=0, header=0).values


import torch
vars_in_single_time_step = (variables*lookback)+2
sequence_length = steps
num_actions = 5
state_dim = (variables*lookback)
action_dim = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


generate_buffer_from_dataset(data, vars_in_single_time_step,
                                 sequence_length, state_dim, action_dim,
                                 device, env=dataset, seed=0, buffer_name='b1'
                                 )

full_buffer_name = f"b1_{dataset}_0"


import utils
replay_buffer = utils.StandardBuffer(state_dim,
                                         action_dim,
                                         1000,
                                         10000000,
                                         device
                                         )
replay_buffer.load(f"./buffers/{full_buffer_name}")


state, action, next_state, reward, done = replay_buffer.sample()
state.shape


from main import *

ddargs = dargs(num_actions, 
               state_dim, 
               action_dim, 
               env=dataset, 
               seed=0, 
               buffer_name='b1', 
               optimizer="Adam", 
               optimizer_parameters={"lr": 3e-4}, 
               do_eval_policy = False,  eval_freq=1e4, 
               max_timesteps=1e6, 
               env_made=None)


import datetime
a = datetime.datetime.now()
model = train_DBCQ(ddargs, device)
b = datetime.datetime.now()


print((b-a).total_seconds())



import pickle
file = open(f"{dataset}_{samples}x{steps}_BCQ.pkle", 'wb')
pickle.dump(model, file)



