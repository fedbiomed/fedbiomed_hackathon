#!/usr/bin/env python
# coding: utf-8

# # Fedbiomed Researcher

# Use for developing (autoreloads changes made across packages)


# ## Setting the client up
# It is necessary to previously configure a node:
# 1. `./scripts/fedbiomed_run node add`
#   * Select option 2 (default) to add MNIST to the client
#   * Confirm default tags by hitting "y" and ENTER
#   * Pick the folder where MNIST is downloaded (this is due torch issue https://github.com/pytorch/vision/issues/3549)
#   * Data must have been added (if you get a warning saying that data must be unique is because it's been already added)
#
# 2. Check that your data has been added by executing `./scripts/fedbiomed_run node add`
# 3. Run the node using `./scripts/fedbiomed_run node add`. Wait until you get `Connected with result code 0`. it means you are online.


# ## Create an experiment to train a model on the data found



# Declare a torch.nn MyTrainingPlan class to send for training on the node

import torch
import torch.nn as nn
from fedbiomed.common.torchnn import TorchTrainingPlan
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# you can use any class name eg:
# class AlterTrainingPlan(TorchTrainingPlan):
class MyTrainingPlan(TorchTrainingPlan):
    def __init__(self):
        super(MyTrainingPlan, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        # Here we define the custom dependencies that will be needed by our custom Dataloader
        # In this case, we need the torch DataLoader classes
        # Since we will train on MNIST, we need datasets and transform from torchvision
        deps = ["from torchvision import datasets, transforms",
               "from torch.utils.data import DataLoader"]
        self.add_dependency(deps)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def training_data(self, batch_size = 48):
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        print("[INFO] Training on dataset: " + str(self.dataset_path))
        dataset1 = datasets.MNIST(self.dataset_path, train=True, download=False, transform=transform)
        train_kwargs = {'batch_size': batch_size, 'shuffle': True}
        data_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        return data_loader

    def training_step(self, data, target):
        output = self.forward(data)
        loss   = torch.nn.functional.nll_loss(output, target)
        return loss


# This group of arguments correspond respectively:
# * `model_args`: a dictionary with the arguments related to the model (e.g. number of layers, features, etc.). This will be passed to the model class on the client side.
# * `training_args`: a dictionary containing the arguments for the training routine (e.g. batch size, learning rate, epochs, etc.). This will be passed to the routine on the client side.
#
# **NOTE:** typos and/or lack of positional (required) arguments will raise error. 🤓

model_args = {}

training_args = {
    'batch_size': 48,
    'lr': 1e-3,
    'epochs': 1,
    'dry_run': False,
    'batch_maxnum': 100  # Fast pass for development : only use ( batch_maxnum * batch_size ) samples
}


#    Define an experiment
#    - search nodes serving data for these `tags`, optionally filter on a list of client ID with `clients`
#    - run a round of local training on nodes with model defined in `model_class` + federation with `aggregator`
#    - run for `rounds` rounds, applying the `client_selection_strategy` between the rounds

from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage

tags =  ['#MNIST', '#dataset']
rounds = 2

exp = Experiment(tags=tags,
                #clients=None,
                model_class=MyTrainingPlan,
                # model_class=AlterTrainingPlan,
                # model_path='/path/to/model_file.py',
                model_args=model_args,
                training_args=training_args,
                rounds=rounds,
                aggregator=FedAverage(),
                client_selection_strategy=None,
                save_breakpoints=True)


# Let's start the experiment.
# By default, this function doesn't stop until all the `rounds` are done for all the clients

exp.run()


# Local training results for each round and each node are available in `exp.training_replies` (index 0 to (`rounds` - 1) ).
# For example you can view the training results for the last round below.
#
# Different timings (in seconds) are reported for each dataset of a node participating in a round :
# - `rtime_training` real time (clock time) spent in the training function on the node
# - 'ptime_training` process time (user and system CPU) spent in the training function on the node
# - `rtime_total` real time (clock time) spent in the researcher between sending the request and handling the response, at the `Job()` layer

print("______________ original training replies_________________")
print("\nList the training rounds : ", exp.training_replies.keys())

print("\nList the clients for the last training round and their timings : ")
round_data = exp.training_replies[rounds - 1].data
for c in range(len(round_data)):
    print("\t- {id} :\
        \n\t\trtime_training={rtraining:.2f} seconds\
        \n\t\tptime_training={ptraining:.2f} seconds\
        \n\t\trtime_total={rtotal:.2f} seconds".format(id = round_data[c]['client_id'],
                rtraining = round_data[c]['timing']['rtime_training'],
                ptraining = round_data[c]['timing']['ptime_training'],
                rtotal = round_data[c]['timing']['rtime_total']))
print('\n')
    

del exp  
# here we simulate the removing of the ongoing experiment
# fret not! we have saved breakpoint, so we can retrieve parameters
# of the experiment using `load_breakpoint` method


loaded_exp = Experiment.load_breakpoint()

print("______________ loaded training replies_________________")
#print("\nList the training rounds : ", loaded_exp.training_replies.keys())

print("\nList the clients for the last training round and their timings : ")
round_data = loaded_exp.training_replies[rounds - 1].data
for c in range(len(round_data)):
    #print(round_data[c])
    print("\t- {id} :\
        \n\t\trtime_training={rtraining:.2f} seconds\
        \n\t\tptime_training={ptraining:.2f} seconds\
        \n\t\trtime_total={rtotal:.2f} seconds".format(id = round_data[c]['client_id'],
                rtraining = round_data[c]['timing']['rtime_training'],
                ptraining = round_data[c]['timing']['ptime_training'],
                rtotal = round_data[c]['timing']['rtime_total']))
print('\n')
    
#print(loaded_exp.training_replies[rounds - 1].dataframe)
loaded_exp.run()



# Federated parameters for each round are available in `exp.aggregated_params` (index 0 to (`rounds` - 1) ).
# For example you can view the federated parameters for the last round of the experiment :

print("\nList the training rounds : ", loaded_exp.aggregated_params.keys())

print("\nAccess the federated params for the last training round : ")
print("\t- params_path: ", loaded_exp.aggregated_params[rounds ]['params_path'])
print("\t- parameter data: ", loaded_exp.aggregated_params[rounds ]['params'].keys())
# ## Optional : searching the data

#from fedbiomed.researcher.requests import Requests
#
#r = Requests()
#data = r.search(tags)
#
#import pandas as pd
#for client_id in data.keys():
#    print('\n','Data for ', client_id, '\n\n', pd.DataFrame(data[client_id]))
#
#print('\n')


