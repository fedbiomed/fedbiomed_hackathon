# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Fed-BioMed Federated Analytics.

This module implements the logic for running audited analytics queries
on the datasets belonging to the federation of Fed-BioMed nodes in an experiment.
"""

from functools import reduce
from typing import Any, Dict, List, Tuple, TypeVar
from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.responses import Responses
from fedbiomed.common.serializer import Serializer


QueryResult = Any  # generic type indicating the result from an analytics query
NodeId = str
TDataset = TypeVar('FedbiomedDataset')  # to be defined
TExperiment = TypeVar('Experiment')  # fedbiomed.researcher.experiment.Experiment
TResponses = TypeVar('Responses')  # fedbiomed.researcher.responses.Responses


def fed_mean(exp_instance: TExperiment, **kwargs) -> QueryResult:
    """
    Computes federated mean.

    Args:
        exp_instance: the instance of the experiment
        kwargs: any keyword arguments as defined by the corresponding `mean` function implemented in the `Dataset`
                class

    Returns:
        Results as implemented in the `Dataset` class.
    """
    return exp_instance._submit_fed_analytics_query(query_type='mean', query_kwargs=kwargs)


def fed_std(exp_instance: TExperiment, **kwargs) -> QueryResult:
    """
    Computes federated standard deviation.

    Args:
        exp_instance: the instance of the experiment
        kwargs: any keyword arguments as defined by the corresponding `std` function implemented in the `Dataset`
                class

    Returns:
        Results as implemented in the `Dataset` class.
    """
    return exp_instance._submit_fed_analytics_query(query_type='std', query_kwargs=kwargs)


def _submit_fed_analytics_query(exp_instance: TExperiment,
                                query_type: str,
                                query_kwargs: dict) -> Tuple[QueryResult, Dict[NodeId, QueryResult]]:
    """Helper function executing one round of communication for executing an analytics query on the nodes.

    Args:
        exp_instance: the instance of the experiment
        query_type: identifier for the name of the analytics function to be executed on the node. The
                    `Dataset` class must implement a function with the same name
        query_kwargs: keyword arguments to be passed to the query function executed on the nodes

    Returns:
        - the aggregated result
        - a dictionary of {node_id: node-specific results}
    """
    # serialize query arguments
    serialized_query_kwargs = Serializer.dumps(query_kwargs).hex()
    # set participating nodes
    exp_instance.job().nodes = exp_instance.job().data.node_ids()
    # setup secagg
    secagg_arguments = exp_instance.secagg_setup()
    # prepare query request
    msg = {
        'researcher_id': environ["ID"],
        'job_id': exp_instance.job().id,
        'command': 'analytics_query',
        'query_type': query_type,
        'query_kwargs': serialized_query_kwargs,
        'training_plan_url': exp_instance.job()._repository_args['training_plan_url'],
        'training_plan_class': exp_instance.job()._repository_args['training_plan_class'],
        'secagg_servkey_id': secagg_arguments.get('secagg_servkey_id'),
        'secagg_biprime_id': secagg_arguments.get('secagg_biprime_id'),
        'secagg_random': secagg_arguments.get('secagg_random'),
        'secagg_clipping_range': secagg_arguments.get('secagg_clipping_range'),
    }
    # send query request to nodes
    for cli in exp_instance.job().nodes:
        msg['dataset_id'] = exp_instance.job().data.data()[cli]['dataset_id']
        exp_instance.job().requests.send_message(msg, cli)
    # collect query results from nodes
    exp_instance._responses_history.append(Responses(list()))
    while exp_instance.job().waiting_for_nodes(exp_instance._responses_history[-1]):
        query_results = exp_instance.job().requests.get_responses(look_for_commands=['analytics_query', 'error'],
                                                          only_successful=False)
        for result in query_results.data():
            result['results'] = Serializer.loads(bytes.fromhex(result['results']))
            exp_instance._responses_history[-1].append(result)
    # parse results
    results = [x['results'] for x in exp_instance._responses_history[-1]]
    # prepare data manager (only static methods from the dataset can be used)
    dataset_class = exp_instance.training_plan().dataset_class
    # aggregate results
    if exp_instance.secagg.active:
        aggregation_result = exp_instance._secure_aggregate(results, dataset_class)
    else:
        aggregation_function = getattr(dataset_class, 'aggregate_' + query_type)
        aggregation_result = aggregation_function(results)
    # combine aggregated and node-specific results
    combined_result = (
        aggregation_result,
        {x['node_id']: x['results'] for x in exp_instance._responses_history[-1].data()}
    )
    # store combined results in history
    exp_instance._aggregation_results_history.append(combined_result)
    return combined_result


def _secure_aggregate(exp_instance: TExperiment,
                      results: List[QueryResult],
                      dataset_class: TDataset) -> QueryResult:
    """Computes secure aggregation of analytics query results from each node.

    !!! warning "Limitations"
        The only supported aggregation method is plain (i.e. unweighted) averaging. Thus secure aggregation will
        only yield correct results if the number of samples on each node is the same, and if averaging is the
        correct aggregation operation for a given query.

    Args:
        exp_instance: the instance of the experiment
        results: the list of query results from each node
        data_manager: the data manager class obtained from the training plan
    """
    # compute average of flattened query results
    flattened = exp_instance.secagg.aggregate(
        round_=1,
        encryption_factors={
            x['node_id']: x['results']['encryption_factor'] for x in exp_instance._responses_history[-1]
        },
        total_sample_size=reduce(
            lambda x,y: x + y['results']['num_samples'],
            exp_instance._responses_history[-1], 0),
        model_params={
            x['node_id']: x['results']['flat'] for x in exp_instance._responses_history[-1]
        }
    )
    # unflatten aggregated results
    unflatten = getattr(dataset_class, 'unflatten')
    return unflatten({
        'flat': flattened,
        'format': results[0]['format']})

def fed_analytics(cls):
    """
    Decorator providing federated analytics API for researcher.

    This decorator defines the public API for the analytics queries that can be run within the Fed-BioMed federation.
    This decorator also defines the logic for "translating" an analytics query request by the researcher into a job
    to be executed on the nodes, and for orchestrating said job.

    !!! info "Researcher interface"
        The intention of this module is to provide additional methods to `Experiment` without cluttering it. Hence,
        it is implemented as a decorator.

    Assumptions:

    - the `Experiment` holds a well-defined [`Job`][fedbiomed.researcher.job.Job] as well a well-defined [`FederatedDataSet`][fedbiomed.researcher.datasets.FederatedDataSet];
    - the dataset class corresponding to the data type on the nodes implements the appropriate analytics functions

    The decorator adds the following attributes to the `Experiment` class.

    Attributes:
       _responses_history (list): a record of all successful query responses from nodes
       _aggregation_results_history (list): a record of all aggregation results

    Adding a new analytics query
    ===

    First, define a `fed_<query>` function (e.g. `fed_mean`) which calls the protected method
    `_submit_fed_analytics_query`. Then, in order to be compliant with our workflow, the Dataset class must
    implement the following functions:

    | name | description | example |
    | --- | --- | --- |
    | `init` | the dataset must support construction without setting a data path | `TabularDataset.__init__` |
    | `<query>` | takes `query_kwargs` as input and returns a dict of serializable items representing the result of the query | [`TabularDataset.mean`][fedbiomed.common.data._tabular_dataset.TabularDataset.mean] |
    | `aggregate_<query>` | static method that takes the results from each node's query and returns the aggregated values | [`TabularDataset.aggregate_mean`][fedbiomed.common.data._tabular_dataset.TabularDataset.aggregate_mean] |

    Algorithm
    ===

    Pseudo-code:

    - when `exp.fed_<query>` is called, the arguments are serialized and an `AnalyticsQueryRequest`
      is sent to the nodes
    - on the node, a `Round` is instantiated and the `run_analytics_query` method is called
    - each node instantiates the `TrainingPlan` and executes the `training_data` function to obtain a `DataManager`
    - from `DataManager` we obtain the `Dataset` object, on which `Round` calls `Dataset.<query>` to obtain the
      node-specific query results. The format of the results is defined by the `Dataset` class itself
    - The node-specific results are serialized and sent to the aggregator, which deserializes them
    - the aggregator also instantiates the `TrainingPlan` and calls `training_data`. However, the dataset object
      is not linked to any path. We only use static methods from the dataset class on the aggregator
    - the `Dataset.aggregate_<query>` static method is called

    Secure Aggregation
    ===

    Federated analytics queries may support the secure aggregation protocol.
    This will prevent the researcher from accessing the query results from each node, as they will only be able to
    view in clear the aggregated result.

    !!! warning "Limitations"
        The only supported aggregation method is plain (i.e. unweighted) averaging. Thus secure aggregation will only
        yield correct results if the number of samples on each node is the same, and if averaging is the correct
        aggregation operation for a given query.

    To support secure aggregation, the `Dataset` class must also implement:

    | name | description | example |
    | --- | --- | --- |
    | `flatten_<query>` | optional static method that takes the output of <query> and returns a dict with at least two keys: `flat` corresponding to a flattened (1-dimensional) array of results, and `format` with the necessary shape to unflatten the array | [`TabularDataset.flatten_mean`][fedbiomed.common.data._tabular_dataset.TabularDataset.flatten_mean] |
    | `unflatten_<query>` | optional static method that takes a flattened output (i.e. a dict with the `flat` and `format` keys) and returns the reconstructed results | [`TabularDataset.unflatten_mean`][fedbiomed.common.data._tabular_dataset.TabularDataset.unflatten_mean] |

    The algorithm for executing the federated query is changed as follows in the case of secure aggregation:

    - when `exp.fed_<query>` is called, the arguments are serialized and an `AnalyticsQueryRequest`
      is sent to the nodes
    - on the node, a `Round` is instantiated and the `run_analytics_query` method is called
    - each node instantiates the `TrainingPlan` and executes the `training_data` function to obtain a `DataManager`
    - from `DataManager` we obtain the `Dataset` object, on which `Round` calls `Dataset.<query>` to obtain the
      node-specific query results. The format of the results is defined by the `Dataset` class itself
    - The node-specific results are flattened and then serialized, before being sent to the aggregator,
      which deserializes them
    - the aggregator also instantiates the `TrainingPlan` and calls `training_data`. However, the dataset object
      is not linked to any path. We only use static methods from the dataset class on the aggregator
    - the secure aggregation method is called. This method only computes the sum of the flattened parameters from each
      node's contribution
    - the summed encrypted weights are decrypted and averaged (unweighted). Finally, the results are unflattened

    """
    # Additional attributes for Experiment class
    cls._responses_history: List[TResponses] = list()
    cls._aggregation_results_history: List[Tuple[QueryResult, Dict[NodeId, QueryResult]]] = list()
    # Analytics Public API
    cls.fed_mean = fed_mean
    cls.fed_std = fed_std
    # Analytics private methods
    cls._submit_fed_analytics_query = _submit_fed_analytics_query
    cls._secure_aggregate = _secure_aggregate
    return cls
