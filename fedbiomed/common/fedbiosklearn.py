import inspect
from joblib import dump, load
import torch
import numpy as np
from sklearn.linear_model import SGDRegressor
import json

class GeneralSkLearnModel():

    ''' Set scikit learning model here. '''
    def initialize(self, model, init_parameters):
        self.model = model
        self.parameters_name = init_parameters.keys()
        self.initialize_model(init_parameters)    

    ''' Must prescribe how to initialize the model's parameters'''
    def initialize_model(self, parameters):
        for params in self.parameters_name:
            setattr(self.model, params, parameters[params])

    ''' Provide partial fit method of scikit learning model here. '''
    def partial_fit(self,X,y):
        pass

    '''Perform in this method all data reading and data transformations you need.
    At the end you should provide a couple (X,y) as indicated in the partial_fit
    method of the scikit learn class.'''
    def training_data(self, batch_size=None):
        pass

    ''' Provide a dictionnary with the parameters you need to be fitted, refer to
     scikit documentation for a detail of parameters '''
    def after_training_params(self):
        return {value:key for value in self.parameters_name for key in getattr(self.model[value])}

    '''
    Method training_routine called in Round, to change only if you know what you are doing.
    '''
    def training_routine(self, epochs=1, log_interval=10, lr=1e-3, batch_size=50, batch_maxnum=0, dry_run=False,
                         logger=None):
        print('SGD Regressor training batch size ', batch_size)
        (data, target) = self.training_data(batch_size=batch_size)
        for r in range(epochs):
            # do not take into account more than batch_maxnum batches from the dataset
            if batch_maxnum == 0 :
                self.reg.partial_fit(data,target)
            else:
                print('Not yet implemented batch_maxnum != 0')

    def __init__(self, kwargs):
        self.batch_size = 100
        self.dependencies = [   "from fedbiomed.common.fedbiosklearn import SkLearnModel",
                                "import inspect",
                                "import pickle",
                                "import numpy as np",
                                "import pandas as pd",

                                "from torchvision import datasets, transforms",
                             ]
        self.initialize(kwargs['model'], kwargs['init_parameters'])

        self.dataset_path = None
        params_sgd = self.model.get_params()
        from_kwargs_sgd_proper_pars = {key: kwargs[key] for key in kwargs if key in params_sgd}
        params_sgd.update(from_kwargs_sgd_proper_pars)
        self.model.set_params(**params_sgd)

    # provided by fedbiomed // necessary to save the model code into a file
    def add_dependency(self, dep):
        self.dependencies.extend(dep)
        pass

    '''Save the code to send to nodes '''
    def save_code(self, filename: str):
        """Save the class code for this training plan to a file
                Args:
                    filename (string): path to the destination file

                Returns:
                    None

                Exceptions:
                    none
        """
        content = ""
        for s in self.dependencies:
            content += s + "\n"

        content += "\n"
        content += inspect.getsource(self.__class__)

        # try/except todo
        file = open(filename, "w")
        file.write(content)
        file.close()

    ''' Save method for parameter communication, internally is used
    dump and load joblib library methods '''
    def save(self, filename, params: dict=None):
        '''
        Save can be called from Job or Round.
            From round is always called with params.
            From job is called with no params in constructor and
            with params in update_parameters.

            Torch state_dict has a model_params object. model_params tag
            is used in the code. This is why this tag is
            used in sklearn case.
        '''
        file = open(filename, "wb")
        if params is None:
            dump(self.model, file)
        else:
            if params.get('model_params') is not None: # called in the Round
                for p in self.parameters_name:
                    setattr(self.model, p, params['model_params'][p])
                #self.model.coef_ = params['model_params']['coef_']
                #self.model.intercept_ = params['model_params']['intercept_']
            else:
                for p in self.parameters_name:
                    setattr(self.model, p, params['p'])
                #self.model.coef_ = params['coef_']
                #self.model.intercept_ = params['intercept_']
            dump(self.model, file)
        file.close()

    ''' Save method for parameter communication, internally is used
        dump and load joblib library methods '''
    def load(self, filename, to_params: bool = False):
        '''
        Load can be called from Job or Round.
            From round is called with no params
            From job is called with  params
        '''
        di_ret = {}
        file = open( filename , "rb")
        if not to_params:
            self.model = load(file)
            di_ret =  self.model
        else:
            self.model =  load(file)
            di_ret['model_params'] = {value:key for value in self.parameters_name for key in getattr(self.model[value])}  #{'coef_': self.model.coef_,'intercept_':self.model.intercept_}
        file.close()
        return di_ret

    # provided by the fedbiomed / can be overloaded // need WORK
    def logger(self, msg, batch_index, log_interval = 10):
        pass

    # provided by the fedbiomed // should be moved in a DATA manipulation module
    def set_dataset(self, dataset_path):
        self.dataset_path = dataset_path
        print('Dataset_path',self.dataset_path)

    def add_dependency(self, dep):
        self.dependencies.extend(dep)
        pass

    def get_model(self):
        return self.model




class SkLearnModel():

    ''' Provide partial fit method of scikit learning model here. '''
    def partial_fit(self,X,y):
        pass

    '''Perform in this method all data reading and data transformations you need.
    At the end you should provide a couple (X,y) as indicated in the partial_fit
    method of the scikit learn class.'''
    def training_data(self, batch_size=None):
        pass

    ''' Provide a dictionnary with the parameters you need to be fitted, refer to
     scikit documentation for a detail of parameters '''
    def after_training_params(self):
        pass

    '''
    Method training_routine called in Round, to change only if you know what you are doing.
    '''
    def training_routine(self, epochs=1, log_interval=10, lr=1e-3, batch_size=50, batch_maxnum=0, dry_run=False,
                         logger=None):
        print('SGD Regressor training batch size ', batch_size)
        (data, target) = self.training_data(batch_size=batch_size)
        for r in range(epochs):
            # do not take into account more than batch_maxnum batches from the dataset
            if batch_maxnum == 0 :
                self.reg.partial_fit(data,target)
            else:
                print('Not yet implemented batch_maxnum != 0')

    def __init__(self,kwargs):
        self.batch_size = 100
        self.dependencies = [   "from fedbiomed.common.fedbiosklearn import SkLearnModel",
                                "import inspect",
                                "import pickle",
                                "import numpy as np",
                                "import pandas as pd",

                                "from torchvision import datasets, transforms",
                             ]
        self.dataset_path = None
        params_sgd = SGDRegressor().get_params()
        from_kwargs_sgd_proper_pars = {key: kwargs[key] for key in kwargs if key in params_sgd}
        params_sgd.update(from_kwargs_sgd_proper_pars)
        self.reg = SGDRegressor()
        self.reg.set_params(**params_sgd)
        self.reg.coef_ =  np.zeros(kwargs['number_columns'])
        self.reg.intercept_ = [0.]

    # provided by fedbiomed // necessary to save the model code into a file
    def add_dependency(self, dep):
        self.dependencies.extend(dep)
        pass

    '''Save the code to send to nodes '''
    def save_code(self, filename: str):
        """Save the class code for this training plan to a file
                Args:
                    filename (string): path to the destination file

                Returns:
                    None

                Exceptions:
                    none
        """
        content = ""
        for s in self.dependencies:
            content += s + "\n"

        content += "\n"
        content += inspect.getsource(self.__class__)

        # try/except todo
        file = open(filename, "w")
        file.write(content)
        file.close()

    ''' Save method for parameter communication, internally is used
    dump and load joblib library methods '''
    def save(self, filename, params: dict=None):
        '''
        Save can be called from Job or Round.
            From round is always called with params.
            From job is called with no params in constructor and
            with params in update_parameters.

            Torch state_dict has a model_params object. model_params tag
            is used in the code. This is why this tag is
            used in sklearn case.
        '''
        file = open(filename, "wb")
        if params is None:
            dump(self.reg, file)
        else:
            if params.get('model_params') is not None: # called in the Round
                self.reg.coef_ = params['model_params']['coef_']
                self.reg.intercept_ = params['model_params']['intercept_']
            else:
                self.reg.coef_ = params['coef_']
                self.reg.intercept_ = params['intercept_']
            dump(self.reg, file)
        file.close()

    ''' Save method for parameter communication, internally is used
        dump and load joblib library methods '''
    def load(self, filename, to_params: bool = False):
        '''
        Load can be called from Job or Round.
            From round is called with no params
            From job is called with  params
        '''
        di_ret = {}
        file = open( filename , "rb")
        if not to_params:
            self.reg = load(file)
            di_ret =  self.reg
        else:
            self.reg =  load(file)
            di_ret['model_params'] = {'coef_': self.reg.coef_,'intercept_':self.reg.intercept_}
        file.close()
        return di_ret

    # provided by the fedbiomed / can be overloaded // need WORK
    def logger(self, msg, batch_index, log_interval = 10):
        pass

    # provided by the fedbiomed // should be moved in a DATA manipulation module
    def set_dataset(self, dataset_path):
        self.dataset_path = dataset_path
        print('Dataset_path',self.dataset_path)

    def add_dependency(self, dep):
        self.dependencies.extend(dep)
        pass

    def get_model(self):
        return self.reg
