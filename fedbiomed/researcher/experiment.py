import logging
import os
import json
import inspect
from typing import Callable, Union, Dict, Any, TypeVar, Type

from fedbiomed.common.logger import logger
from fedbiomed.common.constants import ErrorNumbers
from fedbiomed.researcher.environ import environ
from fedbiomed.common.fedbiosklearn import SGDSkLearnModel
from fedbiomed.common.torchnn import TorchTrainingPlan

from fedbiomed.researcher.filetools import create_exp_folder, choose_bkpt_file, \
    create_unique_link, create_unique_file_link, find_breakpoint_path
from fedbiomed.researcher.aggregators import fedavg, aggregator
from fedbiomed.researcher.strategies.strategy import Strategy
from fedbiomed.researcher.strategies.default_strategy import DefaultStrategy
from fedbiomed.researcher.requests import Requests
from fedbiomed.researcher.job import Job
from fedbiomed.researcher.datasets import FederatedDataSet
from fedbiomed.researcher.monitor import Monitor

_E = TypeVar("Experiment")  # only for typing


class Experiment(object):
    """
    This class represents the orchestrator managing the federated training
    """

    def __init__(self,
                 tags: list = None,
                 nodes: list = None,
                 model_class: Union[Type[Callable], Callable] = None,
                 model_path: str = None,
                 model_args: dict = {},
                 training_args: dict = None,
                 rounds: int = 1,
                 aggregator: Union[Type[aggregator.Aggregator], aggregator.Aggregator] = None,
                 node_selection_strategy: Union[Type[Strategy], Strategy] = None,
                 save_breakpoints: bool = False,
                 training_data: Union[dict, FederatedDataSet] = None,
                 tensorboard: bool = False,
                 experimentation_folder: str = None
                 ):

        """ Constructor of the class.


        Args:
            tags (tuple): tuple of string with data tags
            nodes (list, optional): list of node_ids to filter the nodes
                                    to be involved in the experiment.
                                    Defaults to None (no filtering).
            model_class (Union[Type[Callable], Callable], optional): name or
                                    instance (object) of the model class to use
                                    for training.
                                    Should be a str type when using jupyter notebook
                                    or a Callable when using a simple python
                                    script.
            model_path (string, optional) : path to file containing model code
            model_args (dict, optional): contains output and input feature
                                        dimension. Defaults to None.
            training_args (dict, optional): contains training parameters:
                                            lr, epochs, batch_size...
                                            Defaults to None.
            rounds (int, optional): the number of communication rounds
                                    (nodes <-> central server).
                                    Defaults to 1.
            aggregator (Union[Type[aggregator.Aggregator], aggregator.Aggregator], optional):
                                    class or object defining the method
                                    for aggregating local updates.
                                    Default to None (uses fedavg.FedAverage() for training)
            node_selection_strategy (Union[Type[Strategy], Strategy], optional):
                                    class or object defining how nodes are sampled at each round
                                    for training, and how non-responding nodes are managed.
                                    Defaults to None (uses DefaultStrategy for training)
            save_breakpoints (bool, optional): whether to save breakpoints or
                                                not. Breakpoints can be used
                                                for resuming a crashed
                                                experiment. Defaults to False.
            training_data (Union [dict, FederatedDataSet], optional):
                    FederatedDataSet object or
                    dict of the node_id of nodes providing datasets for the experiment,
                    datasets for a node_id are described as a list of dict, one dict per dataset.
                    Defaults to None, datasets are searched from `tags` and `nodes`.
            tensorboard (bool): Tensorboard flag for displaying scalar values
                                during training in every node. If it is true,
                                monitor will write scalar logs into
                                `./runs` directory.
            experimentation_folder (str, optional): choose a specific name for the
                    folder where experimentation result files and breakpoints are stored.
                    This should just contain the name for the folder not a path.
                    The name is used as a subdirectory of `environ[EXPERIMENTS_DIR])`.
                    - Caveat : if using a specific name this experimentation will not be
                    automatically detected as the last experimentation by `load_breakpoint`
                    - Caveat : do not use a `experimentation_folder` name finishing
                    with numbers ([0-9]+) as this would confuse the last experimentation
                    detection heuristic by `load_breakpoint`.
        """

        if tags:
            # verify that tags is a list, force a list if a simple string is provided (for convenience)
            # raise an error if not
            tags = [tags] if isinstance(tags, str) else tags
            if not isinstance(tags, list):
                logger.critical("The argument `tags` should be a list of string or string")
                return False

            self._tags = tags
        else:
            self._tags = tags

        self._nodes = nodes
        self._reqs = Requests()

        if training_data:
            if not isinstance(training_data, FederatedDataSet) and isinstance(training_data, dict):
                # TODO: Check dict has proper schema
                self._fds = FederatedDataSet(training_data)
                logger.info('Training data has been provided, search request will be ignored.')
            else:
                logger.critical('Training data is not a type of FederatedDataset or Dict')
                return

        elif self._tags:
            self._fds = FederatedDataSet(self._reqs.search(self._tags, self._nodes))
        else:
            self._fds = None

        # Initialize aggregator if it is provided
        if aggregator is None:
            self._aggregator = fedavg.FedAverage()
        else:
            self._aggregator = aggregator

        # Initialize node selection strategy
        if node_selection_strategy is None and self._fds:
            self._node_selection_strategy = DefaultStrategy(self._fds)
        elif node_selection_strategy and self._fds:
            if inspect.isclass(self._node_selection_strategy):
                self._node_selection_strategy = node_selection_strategy(self._fds)
            else:
                logger.critical("`node_selection_strategy should be class`")
        else:
            self._node_selection_strategy = None

        self._round_init = 0  # start from round 0
        self._round_current = 0
        self._aggregator = aggregator

        self._experimentation_folder = create_exp_folder(experimentation_folder)

        self._model_class = model_class
        self._model_path = model_path
        self._model_args = model_args
        self._training_args = training_args
        self._rounds = rounds

        status, _ = self._before_job_init()
        if status:
            self._job = Job(reqs=self._reqs,
                            model=self._model_class,
                            model_path=self._model_path,
                            model_args=self._model_args,
                            training_args=self._training_args,
                            data=self._fds,
                            keep_files_dir=self.experimentation_path)
        else:
            self._job = None

        # structure (dict ?) for additional parameters to the strategy
        # currently unused, to be defined when needed
        # self._sampled = None

        self._aggregated_params = {}
        self._save_breakpoints = save_breakpoints

        #  Monitoring loss values with tensorboard
        if tensorboard:
            self._monitor = Monitor()
            self._reqs.add_monitor_callback(self._monitor.on_message_handler)
        else:
            self._monitor = None
            # Remove callback. Since reqeust class is singleton callback
            # function might be already added into request before.
            self._reqs.remove_monitor_callback()

    # Getters ---------------------------------------------------------------------------------------------------------

    def training_replies(self):
        return self._job.training_replies

    def aggregated_params(self):
        return self._aggregated_params

    def job(self):
        return self._job

    def model_instance(self):
        return self._job.model

    def tags(self):
        return self._tags

    def model_args(self):
        return self._model_args

    def training_args(self):
        return self._training_args

    def model_path(self):
        return self._model_path

    def model_class(self):
        return self._model_class

    def aggregator(self):
        return self._aggregator

    def node_selection_strategy(self):
        return self._node_selection_strategy

    def nodes(self):
        return self._nodes

    def training_data(self):
        return self._fds

    def monitor(self):
        return self._monitor

    def rounds(self):
        return self._rounds

    def experimentation_folder(self):
        return self._experimentation_folder

    def experimentation_path(self):
        return os.path.join(environ['EXPERIMENTS_DIR'], self._experimentation_folder)

    def breakpoint(self):
        return self._save_breakpoints

    # -----------------------------------------------------------------------------------------------------------------

    # Setters ---------------------------------------------------------------------------------------------------------

    def set_model_args(self, model_args: Dict):
        """ Setter for Model Arguments. This method should also update/set model arguments in
        Job object.

        Args:
            model_args (dict): Model arguments
        """

        # TODO: Job uses model arguments in init method for building TrainingPlan (Model Class).
        # After Job has initialized setting new model arguments will require to reinitialize the job.
        # Job needs to be refactored to avoid rebuild after the arguments have changed.
        self._model_args = model_args
        logger.info('Model arguments has been changed, please recreate the Job by running `.set_job()`')

        return True

    def set_training_args(self, training_args):

        """ Setter for training arguments. Updates the Job object with new
            training arguments.

        Args:
            training_args (dict): Training arguments
        """
        self._training_args = training_args

        # Update training arguments if job is already initialized
        if self._job:
            self._job._training_args = training_args

        return True

    def set_tags(self, tags: Union[tuple, str]):
        """ Setter for tags. Since tags are the main criteria for selecting node based on
            dataset, this method sends search request to node to check if they have the
            dataset or not.
            Args:
                tags (str | List): List of tags or single string tag.
        """
        self._tags = [tags] if isinstance(tags, str) else tags
        if not isinstance(self._tags, list):
            logger.critical("experiment parameter tags is not a string list or string list")
            return False

        return True

    def set_breakpoints(self, save_breakpoints: bool = True):
        self._save_breakpoints = save_breakpoints
        return True

    def set_training_data(self,
                          tags: list = None,
                          nodes: list = None,
                          training_data: Union[dict, FederatedDataSet] = None):
        """ Setting training data for federated training.

        """

        # Verify tags if it is provided and update self._tags
        if tags:
            tags = [tags] if isinstance(tags, str) else tags
            if isinstance(self._tags, list):
                self._tags = tags
            else:
                logger.error("The argument `tags` should be a list of string or string")
                return False

        # Update nodes if it is provided
        if nodes:
            if not isinstance(self._tags, list):
                logger.error("The argument `nodes` should be list of node ids")
                return False

            self._nodes = nodes

        if training_data:
            if not isinstance(training_data, FederatedDataSet) and isinstance(training_data, dict):
                # TODO: Check dict has proper schema
                self._fds = FederatedDataSet(training_data)
                logger.info('Training data has been provided, search request will be ignored')
            else:
                logger.error('Training data is not a type of FederatedDataset or Dict')
                return False

        elif self._tags:
            self._fds = self._reqs.search(self._tags, self._nodes)
        else:
            logger.error('Either provide tags or FederatedDataset')
            return False

        return True

    def set_job(self):

        status, messages = self._before_job_init()
        if status:
            self._job = Job(reqs=self._reqs,
                            model=self._model_class,
                            model_path=self._model_path,
                            model_args=self._model_args,
                            training_args=self._training_args,
                            data=self._fds,
                            keep_files_dir=self.experimentation_path)
            return True
        else:
            raise Exception('Error while setting Job: \n\n \t   %s' % '\n'.join(messages))

    # -----------------------------------------------------------------------------------------------------------------

    # PROPOSAL: OLD property methods. We can keep them and raise warning about they are deprecated  and ---------------
    # REMOVE THEM in the version v3.5
    @property
    def training_replies(self):
        # TODO: Remove this method in v3.5
        logger.warning('Calling "Experiment.training_replies" as property has been deprecated and '
                       'will be removed in future releases. Please use `Experiment.training_replies()` '
                       'to get `training_replies`.')
        return self._job.training_replies

    @property
    def aggregated_params(self):
        # TODO: Remove this method in v3.5
        logger.warning('Calling "Experiment.aggregated_params" as property has been deprecated and '
                       'will be removed in future releases. Please use `Experiment.aggregated_params()` as method.')
        return self._aggregated_params

    @property
    def model_instance(self):
        # TODO: Remove this method in v3.5
        logger.warning('Calling "Experiment.model_instance" as property is deprecated and '
                       'will be removed in future releases. Please use `Experiment.model_instance()` as method.')
        return self._job.model

    @property
    def experimentation_folder(self):
        # TODO: Remove this method in v3.5
        logger.warning('Calling "experimentation_folder" as property is deprecated and '
                       'will be removed in future releases. Please use `experimentation_folder()` as method.')
        return self._experimentation_folder

    @property
    def experimentation_path(self):
        # TODO: Remove this method in v3.5
        logger.warning('Calling "model_instance.experimentation_path" as property is deprecated and '
                       'will be removed in future releases. Please use '
                       '`model_instance.experimentation_path()`.')
        return os.path.join(environ['EXPERIMENTS_DIR'], self._experimentation_folder)

    # -----------------------------------------------------------------------------------------------------------------

    def run_once(self):
        """ Runs the experiment only once. It will increase global round each time
        this method is called

        """
        status, messages = self._before_experiment_run()

        if status:
            # Sample nodes using strategy (if given)
            self._job.nodes = self._node_selection_strategy.sample_nodes(self._round_current)
            logger.info('Sampled nodes in round ' + str(self._round_current) + ' ' + str(self._job.nodes))
            # Trigger training round on sampled nodes
            answering_nodes = self._job.start_nodes_training_round(round=self._round_current)

            # refining/normalizing model weigths received from nodes
            model_params, weights = self._node_selection_strategy.refine(
                self._job.training_replies[self._round_current], self._round_current)

            # aggregate model from nodes to a global model
            aggregated_params = self._aggregator.aggregate(model_params,
                                                           weights)
            # write results of the aggregated model in a temp file
            aggregated_params_path = self._job.update_parameters(aggregated_params)
            logger.info(f'Saved aggregated params for round {self._round_current} in {aggregated_params_path}')

            self._aggregated_params[self._round_current] = {'params': aggregated_params,
                                                            'params_path': aggregated_params_path}
            if self._save_breakpoints:
                self._save_breakpoint(self._round_current)

            if self._monitor is not None:
                # Close SummaryWriters for tensorboard
                self._monitor.close_writer()

            self._round_current += 1
        else:
            raise Exception('Error while running the experiment: \n\n \t   %s' % '\n'.join(messages))

        pass

    def run(self, rounds: int = None):
        """Runs an experiment, ie trains a model on nodes for a
        given number of rounds.
        It involves the following steps:


        Args:
            rounds (int, optional): Number of round that the experiment will run
        Raises:
            NotImplementedError: [description]
        Returns:
            None

        """

        # Run experiment
        if self._round_init >= self._rounds:
            logger.info("Round limit reached. Nothing to do")
            return

        # Find out how many rounds wil be run
        rounds_to_run = rounds if rounds else self._rounds

        for _ in range(rounds_to_run):
            # Run ->
            self.run_once()
            # Increase round state
            self._round_current += 1

    def model_file(self, display: bool = True):

        """ This method displays saved final model for the experiment
            that will be send to the nodes for training.
        """
        model_file = self._job.model_file

        # Display content so researcher can copy
        if display:
            with open(model_file) as file:
                content = file.read()
                file.close()
                print(content)

        return self._job.model_file

    def check_model_status(self):

        """ Method for checking model status whether it is approved or
            not by the nodes
        """
        responses = self._job.check_model_is_approved_by_nodes()

        return responses

    def info(self):
        """ Information about status of the current experiment. Method  lists all the
        parameters/arguments of the experiment and inform user about the
        can the experiment be run.

        Returns:
            dict : {key (experiment argument) : value }
        """

        pass

    def _before_job_init(self):
        """ This method checks are all the necessary arguments has been set to
        initialize `Job` class.
`
        Returns:
            status, missing_attributes (bool, List)
        """
        no_none_args_msg = {"_training_args": ErrorNumbers.FB410.value,
                            "_fds": ErrorNumbers.FB411.value,
                            '_model_class': ErrorNumbers.FB412.value,
                            }

        return self._argument_controller(no_none_args_msg)

    def _before_experiment_run(self):

        no_none_args_msg = {"_job": ErrorNumbers.FB413.value,
                            "_node_selection_strategy": ErrorNumbers.FB414.value,
                            '_aggregator': ErrorNumbers.FB415.value,
                            }

        return self._argument_controller(no_none_args_msg)

    def _argument_controller(self, arguments: dict):

        messages = []
        for arg, message in arguments.items():
            if arg in self.__dict__ and self.__dict__[arg] is not None:
                continue
            else:
                messages.append(message)
        status = True if len(messages) == 0 else False

        return status, messages

    def _save_breakpoint(self, round: int = 0):
        """
        Saves breakpoint with the state of the training at a current round.
        The following Experiment attributes will be saved:
         - round_number
         - round_number_due
         - tags
         - experimentation_folder
         - aggregator
         - node_selection_strategy
         - training_data
         - training_args
         - model_args
         - model_path
         - model_class
         - aggregated_params
         - job (attributes returned by the Job, aka job state)

        Args:
            - round (int, optional): number of rounds already executed.
              Starts from 0. Defaults to 0.
        """

        breakpoint_path, breakpoint_file_name = \
            choose_bkpt_file(self._experimentation_folder, round)

        state = {
            'training_data': self._fds.data(),
            'training_args': self._training_args,
            'model_args': self._model_args,
            'model_path': self._job.model_file,  # only in Job we always model saved to a file
            # with current version
            'model_class': self._job.model_class,  # not always available properly
            # formatted in Experiment with current version
            'round_number': round + 1,
            'round_number_due': self._rounds,
            'experimentation_folder': self._experimentation_folder,
            'aggregator': self._aggregator.save_state(),  # aggregator state
            'node_selection_strategy': self._node_selection_strategy.save_state(),
            # strategy state
            'tags': self._tags,
            'aggregated_params': self._save_aggregated_params(
                self._aggregated_params, breakpoint_path),
            'job': self._job.save_state(breakpoint_path)  # job state
        }

        # rewrite paths in breakpoint : use the links in breakpoint directory
        state['model_path'] = create_unique_link(
            breakpoint_path,
            # - Need a file with a restricted characters set in name to be able to import as module
            'model_' + str("{:04d}".format(round)), '.py',
            # - Prefer relative path, eg for using experiment result after
            # experiment in a different tree
            os.path.join('..', os.path.basename(state["model_path"]))
        )

        # save state into a json file.
        breakpoint_file_path = os.path.join(breakpoint_path, breakpoint_file_name)
        with open(breakpoint_file_path, 'w') as bkpt:
            json.dump(state, bkpt)
        logger.info(f"breakpoint for round {round} saved at " + \
                    os.path.dirname(breakpoint_file_path))

    @classmethod
    def load_breakpoint(cls: Type[_E],
                        breakpoint_folder_path: str = None) -> _E:
        """
        Loads breakpoint (provided a breakpoint has been saved)
        so experience can be resumed. Useful if training has crashed
        researcher side or if user wants to resume experiment.

        Args:
            - cls (Type[_E]): Experiment class
            - breakpoint_folder_path (str, optional): path of the breakpoint folder.
              Path can be absolute or relative eg: "var/experiments/Experiment_xx/breakpoints_xx".
              If None, loads latest breakpoint of the latest experiment.
              Defaults to None.

        Returns:
            - _E: Reinitialized experiment. With given object,
              user can then use `.run()` method to pursue model training.
        """

        # get breakpoint folder path (if it is None) and
        # state file
        breakpoint_folder_path, state_file = find_breakpoint_path(breakpoint_folder_path)
        breakpoint_folder_path = os.path.abspath(breakpoint_folder_path)

        # TODO: check if all elements needed for breakpoint are present
        with open(os.path.join(breakpoint_folder_path, state_file), "r") as f:
            saved_state = json.load(f)

        # -----  retrieve breakpoint training data ---
        bkpt_fds = FederatedDataSet(saved_state.get('training_data'))

        # -----  retrieve breakpoint sampling strategy ----
        bkpt_sampling_strategy_args = saved_state.get("node_selection_strategy")
        bkpt_sampling_strategy = cls._create_object(bkpt_sampling_strategy_args, data=bkpt_fds)

        # ----- retrieve federator -----
        bkpt_aggregator_args = saved_state.get("aggregator")
        bkpt_aggregator = cls._create_object(bkpt_aggregator_args)

        # ------ initializing experiment -------

        loaded_exp = cls(tags=saved_state.get('tags'),
                         nodes=None,  # list of previous nodes is contained in training_data
                         model_class=saved_state.get("model_class"),
                         model_path=saved_state.get("model_path"),
                         model_args=saved_state.get("model_args"),
                         training_args=saved_state.get("training_args"),
                         rounds=saved_state.get("round_number_due"),
                         aggregator=bkpt_aggregator,
                         node_selection_strategy=bkpt_sampling_strategy,
                         save_breakpoints=True,
                         training_data=bkpt_fds,
                         experimentation_folder=saved_state.get('experimentation_folder')
                         )

        # ------- changing `Experiment` attributes -------
        loaded_exp._round_init = saved_state.get('round_number')
        loaded_exp._aggregated_params = loaded_exp._load_aggregated_params(
            saved_state.get('aggregated_params'),
            loaded_exp.model_instance.load
        )

        # ------- changing `Job` attributes -------
        loaded_exp._job.load_state(saved_state.get('job'))

        logging.info(f"experimentation reload from {breakpoint_folder_path} successful!")
        return loaded_exp

    @staticmethod
    def _save_aggregated_params(aggregated_params_init: dict, breakpoint_path: str) -> Dict[int, dict]:
        """Extracts and format fields from aggregated_params that need
        to be saved in breakpoint. Creates link to the params file from the `breakpoint_path`
        and use them to reference the params files.

        Args:
            - breakpoint_path (str): path to the directory where breakpoints files
                and links will be saved

        Returns:
            - Dict[int, dict] : extract from `aggregated_params`
        """
        aggregated_params = {}
        for key, value in aggregated_params_init.items():
            params_path = create_unique_file_link(breakpoint_path,
                                                  value.get('params_path'))
            aggregated_params[key] = {'params_path': params_path}

        return aggregated_params

    @staticmethod
    def _load_aggregated_params(aggregated_params: Dict[str, dict], func_load_params: Callable
                                ) -> Dict[int, dict]:
        """Reconstruct experiment results aggregated params structure
        from a breakpoint so that it is identical to a classical `_aggregated_params`

        Args:
            - aggregated_params (Dict[str, dict]) : JSON formatted aggregated_params
              extract from a breakpoint
            - func_load_params (Callable) : function for loading parameters
              from file to aggregated params data structure

        Returns:
            - Dict[int, dict] : reconstructed aggregated params from breakpoint
        """
        # needed for iteration on dict for renaming keys
        keys = [key for key in aggregated_params.keys()]
        # JSON converted all keys from int to string, need to revert
        for key in keys:
            aggregated_params[int(key)] = aggregated_params.pop(key)

        for aggreg in aggregated_params.values():
            aggreg['params'] = func_load_params(aggreg['params_path'], to_params=True)

        return aggregated_params

    # TODO: factorize code with Job and node
    # TODO: add signal handling for error cases
    @staticmethod
    def _create_object(args: Dict[str, Any], **object_kwargs) -> Callable:
        """
        Instantiate a class object from breakpoint arguments.

        Args:
            - args (Dict[str, Any]) : breakpoint definition of a class with `class` (classname),
              `module` (module path) and optional additional parameters containing object state
            - **object_kwargs : optional named arguments for object constructor

        Returns:
            - Callable: object of the class defined by `args` with state restored from breakpoint
        """
        module_class = args.get("class")
        module_path = args.get("module")
        import_str = 'from ' + module_path + ' import ' + module_class

        # import module
        exec(import_str)
        # create a class variable containing the class
        class_code = eval(module_class)
        # instantiate object from module
        if object_kwargs is None:
            object_instance = class_code()
        else:
            object_instance = class_code(**object_kwargs)

        # load breakpoint state for object
        object_instance.load_state(args)

        return object_instance
