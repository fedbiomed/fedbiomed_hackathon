#
# linear inheritance of torch nn.Module
#


import inspect

import torch
import torch.nn as nn

class Torchnn(nn.Module):
    def __init__(self):
        super(Torchnn, self).__init__()

        # cannot use it here !!!! FIXED in training_routine
        #self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        self.optimizer = None

        # data loading // should ne moved to another class
        # self.batch_size = 100
        # self.shuffle    = True

        # training // may be changed in training_routine ??
        self.device = "cpu"

        # list dependencies of the model
        self.dependencies = [ "from fedbiomed.common.torchnn import Torchnn",
                              "import torch",
                              "import torch.nn as nn",
                              "import torch.nn.functional as F",
                             ]

        # to be configured by setters
        self.dataset_path = None


    #################################################
    # provided by fedbiomed
    def training_routine(self, epochs=2, log_interval = 10, lr=1e-3, batch_size=48, batch_maxnum=0, dry_run=False, logger=None):

        if self.optimizer == None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)

        #use_cuda = torch.cuda.is_available()
        #device = torch.device("cuda" if use_cuda else "cpu")
        self.device = "cpu"

        for epoch in range(1, epochs + 1):
            training_data = self.training_data(batch_size=batch_size)
            for batch_idx, (data, target) in enumerate(training_data):
                self.train()
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                res = self.training_step(data, target)
                res.backward()
                self.optimizer.step()

                # do not take into account more than batch_maxnum batches from the dataset
                if (batch_maxnum > 0) and (batch_idx >= batch_maxnum):
                    print('Reached {} batches for this epoch, ignore remaining data'.format(batch_maxnum))
                    break

                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(data),
                        len(training_data.dataset),
                        100 * batch_idx / len(training_data),
                        res.item()))
                    #
                    # deal with the logger here
                    #

                    if dry_run:
                        return

    # provided by fedbiomed // necessary to save the model code into a file
    def add_dependency(self, dep):
        self.dependencies.extend(dep)
        pass

    # provider by fedbiomed
    def save_code(self):

        content = ""
        for s in self.dependencies:
            content += s + "\n"

        content += "\n"
        content += inspect.getsource(self.__class__)

        # try/except todo
        file = open("my_model.py", "w")
        file.write(content)
        file.close()

    # provided by fedbiomed
    def save(self, filename, params: dict=None):
        if params is not None:
            return(torch.save(params, filename))
        else:
            return torch.save(self.state_dict(), filename)

    # provided by fedbiomed
    def load(self, filename, to_params: bool=False):
        if to_params is True:
            return torch.load(filename)
        else:
            return self.load_state_dict(torch.load(filename))

    # provided by the fedbiomed / can be overloaded // need WORK
    def logger(self, msg, batch_index, log_interval = 10):
        pass

    # provided by the fedbiomed // should be moved in a DATA manipulation module
    def set_dataset(self, dataset_path):
        self.dataset_path = dataset_path
        print('Dataset_path',self.dataset_path)

    # provided by the fedbiomed // should be moved in a DATA manipulation module
    def training_data(self, batch_size = 48):

        pass
