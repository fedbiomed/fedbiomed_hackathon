# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

"""
Implements the message exchanges from researcher to nodes
"""

import json
import os
import tabulate
import uuid
import tempfile
from time import sleep
from typing import Any, Dict, Callable, Union, List, Optional, Tuple

from python_minifier import minify

from fedbiomed.common.constants import MessageType
from fedbiomed.common.exceptions import FedbiomedTaskQueueError
from fedbiomed.common.logger import logger
from fedbiomed.common.message import ResearcherMessages, Message
from fedbiomed.common.singleton import SingletonMeta
from fedbiomed.common.tasks_queue import TasksQueue
from fedbiomed.common.training_plans import BaseTrainingPlan
from fedbiomed.common.utils import import_class_object_from_file

from fedbiomed.transport.server import GrpcServer

from fedbiomed.researcher.environ import environ
from fedbiomed.researcher.responses import Responses


class Requests(metaclass=SingletonMeta):
    """
    Represents the requests addressed from Researcher to nodes. It creates a task queue storing reply to each
    incoming message. Starts a message queue and reconfigures  message to be sent into a `Messaging` object.
    """

    def __init__(self):
        """
        Constructor of the class

        Args:
            mess: message to be sent by default.
        """
        # Need to ensure unique per researcher instance message queue to avoid conflicts
        # in case several instances of researcher (with same researcher_id ?) are active,
        # eg: a notebook not quitted and launching a script
        self.queue = TasksQueue(environ['MESSAGES_QUEUE_DIR'] + '_' + str(uuid.uuid4()), environ['TMP_DIR'])

        # defines the sequence used for ping protocol
        self._sequence = 0

        self._monitor_message_callback = None

        # Creates grpc server and starts it
        self._grpc_server = GrpcServer(
            host=environ["SERVER_HOST"],
            port=environ["SERVER_PORT"],
            on_message=self.on_message
        )
        self.start_messaging()


    def start_messaging(self):
        """Start communications endpoint
        """
        self._grpc_server.start()

    def on_message(self, msg: Union[Dict[str, Any], Message], type_: MessageType):
        """ Handler called by the [`ResearcherServer`][fedbiomed.transport.researcher_server] class,
        when a message is received on researcher side.

        It is run in the communication process and must be as quick as possible:
        - it deals with quick messages (eg: ping/pong)
        - it stores the replies of the nodes to the task queue, the message will be
        treated by the main (computing) thread.

        Args:
            msg: de-serialized msg
            type: Reply type one of reply, log, scalar
        """

        if type_ == MessageType.LOG:
            # forward the treatment to node_log_handling() (same thread)
            self.print_node_log_message(msg.get_dict())
        elif type_ == MessageType.REPLY:
            # Adds reply message to the queue
            self.queue.add(ResearcherMessages.format_incoming_message(msg).get_dict())

        elif type_ == MessageType.SCALAR:
            if self._monitor_message_callback is not None:
                # Pass message to Monitor's on message handler
                self._monitor_message_callback(msg.get_dict())
        else:
            logger.error(f"Undefined message type received  {type_} - IGNORING")

    @staticmethod
    def print_node_log_message(log: Dict[str, Any]) -> None:
        """Prints logger messages coming from the node

        It is run on the communication process and must be as quick as possible:
        - all logs (coming from the nodes) are forwarded to the researcher logger
        (immediate display on console/file/whatever)
        """

        # log contains the original message sent by the node
        # FIXME: we should use `fedbiomed.common.json.deserialize` method
        # instead of the json method when extracting json message
        original_msg = json.loads(log["msg"])

        # Loging fancy feedback for training
        logger.info("\033[1m{}\033[0m\n"
                    "\t\t\t\t\t\033[1m NODE\033[0m {}\n"
                    "\t\t\t\t\t\033[1m MESSAGE:\033[0m {}\033[0m\n"
                    "{}".format(log["level"],
                                log["node_id"],
                                original_msg["message"],
                                5 * "-------------"))

    def _add_sequence(self) -> int:
        """Increases sequence number and saves it each time it is called

        Returns:
            Current sequence number
        """
        seq = self._sequence
        self._sequence = + 1

        return seq

    def send_message(self, msg: dict, client: str, add_sequence: bool = False) -> Optional[int]:
        """Request the communication layer to send a message to a node.

        Args:
            msg: the message to send to node
            client: unique node ID of the destination node for the message
            add_sequence: if `True`, add unique sequence number to the message

        Returns:
            sequence: If `add_sequence` is True return the sequence number added to the message.
                If `add_sequence` is False, return None
        """

        if add_sequence:
            msg['sequence'] = self._add_sequence()

        # Send message to client
        self._grpc_server.send(ResearcherMessages.format_outgoing_message(msg), client)

        return msg.get('sequence', None)

    def broadcast(self, message, add_sequence: bool = False) -> Optional[int]:
        """Request the communication layer to roadcast a message to all nodes.

        Args:
            msg: the message to send to nodes
            add_sequence: if `True`, add unique sequence number to the message

        Returns:
            If `add_sequence` is True return the sequence number added to the message.
                If `add_sequence` is False, return None
        """

        if add_sequence:
            message['sequence'] = self._add_sequence()

        # TODO: Return also  the list of node that the messages are sent
        self._grpc_server.broadcast(
            ResearcherMessages.format_outgoing_message(message)
        )

        return message.get('sequence', None)

    def get_messages(self, commands: list = [], time: float = .0) -> Responses:
        """Goes through the queue and gets messages with the specific command

        Args:
            commands: Checks if message is containing the expecting command (the message  is discarded if it doesn't).
                Defaults to None (no command message checking, meaning all incoming messages are considered).
            time: Time to sleep in seconds before considering incoming messages. Defaults to .0.

        Returns:
            Contains the corresponding answers
        """
        sleep(time)

        answers = []
        for _ in range(self.queue.qsize()):
            try:
                item = self.queue.get(block=False)
                self.queue.task_done()

                if not commands or \
                        ('command' in item.keys() and item['command'] in commands):
                    answers.append(item)
                else:
                    # currently trash all other messages
                    pass

            except FedbiomedTaskQueueError:
                # may happend on self.queue.get()
                # if queue is empty -> we ignore it
                pass

        return Responses(answers)

    def get_responses(
            self,
            look_for_commands: list,
            timeout: float = None,
            only_successful: bool = True,
            while_responses: bool = True,
    ) -> Responses:
        """Waits for all nodes' answers, regarding a specific command returns the list of all nodes answers

        Args:
            look_for_commands: instruction that has been sent to node (see `Message` commands)
            timeout: wait for a specific duration before collecting nodes messages. Defaults to None. If set to None;
                uses value in global variable TIMEOUT instead.
            only_successful: deal only with messages that have been tagged as successful (ie with field `success=True`).
                Defaults to True.
            while_responses: if `True`, continue while we get at least one response every
                `timeout` seconds. If False, always terminate after `timeout` even if we get some
                response.
        """
        timeout = timeout or environ['TIMEOUT']
        responses = []

        while True:
            sleep(timeout)
            new_responses = []
            for resp in self.get_messages(commands=look_for_commands, time=0):

                try:
                    if not only_successful:
                        new_responses.append(resp)
                    elif resp['success']:
                        # TODO: test if 'success'key exists
                        # what do we do if not ?
                        new_responses.append(resp)
                except Exception as e:
                    logger.error(f"Incorrect message received: {resp} - error: {e}")
                    pass

            if len(new_responses) == 0:
                "Timeout finished"
                break
            responses += new_responses
            if not while_responses:
                break

        return Responses(responses)

    def ping_nodes(self) -> list:
        """ Pings online nodes

        Returns:
            List ids of up and running nodes
        """

        # Broadcasts ping request
        self.broadcast({
                'researcher_id': environ["ID"],
                'sequence': self._sequence,
                'command': "ping"
            },
            add_sequence=True
        )

        # TODO: check sequence number in pong
        # TODO: (below, above) handle exceptions
        nodes_online = [resp['node_id'] for resp in self.get_responses(look_for_commands=['pong'])]
        return nodes_online

    def search(self, tags: tuple, nodes: Optional[list] = None) -> dict:
        """Searches available data by tags

        Args:
            tags: Tuple containing tags associated to the data researcher is looking for.
            nodes: optionally filter nodes with this list. Default is None, no filtering, consider all nodes

        Returns:
            A dict with node_id as keys, and list of dicts describing available data as values
        """

        # Search datasets based on node specifications
        if nodes is not None:
            logger.info(f'Searching dataset with data tags: {tags} on specified nodes: {nodes}')
            for node in nodes:
                self.send_message(
                    {'tags': tags,
                     'researcher_id': environ['RESEARCHER_ID'],
                     "command": "search"},
                    client=node)
        else:
            logger.info(f'Searching dataset with data tags: {tags} for all nodes')
            # TODO: Unlike MQTT implementation, in gRPC, all the nodes that are broadcasted
            # are known by the researcher gRPC server. Therefore, using timeout is not necessary.
            self.broadcast({'tags': tags,
                            'researcher_id': environ['RESEARCHER_ID'],
                            "command": "search"})

        # TODO: currently a node with no dataset does not answer
        # We may robustify by having any node answer and check this matches the list of ACTIVE nodes
        data_found = {}
        for resp in self.get_responses(look_for_commands=['search']):
            if not nodes:
                data_found[resp.get('node_id')] = resp.get('databases')
            elif resp.get('node_id') in nodes:
                data_found[resp.get('node_id')] = resp.get('databases')

            logger.info('Node selected for training -> {}'.format(resp.get('node_id')))

        if not data_found:
            logger.info("No available dataset has found in nodes with tags: {}".format(tags))

        return data_found

    def list(self, nodes: Optional[list] = None, verbose: bool = False) -> dict:
        """Lists available data in each node

        Args:
            nodes: optionally filter nodes with this list. Default is None, no filtering, consider all nodes
            verbose: If it is true it prints datasets in readable format

        Returns:
            A dict with node_id as keys, and list of dicts describing available data as values
        """

        # If nodes list is provided
        if nodes is not None:
            for node in nodes:
                self.send_message(
                    {"researcher_id": environ['RESEARCHER_ID'],
                     "command": "list"},
                    client=node)
            logger.info(f'Listing datasets of given list of nodes : {nodes}')
        else:
            self.broadcast({'researcher_id': environ['RESEARCHER_ID'],
                            "command": "list"})
            logger.info(f'Listing available datasets in all nodes... {nodes} ')

        # Get datasets from node responses
        data_found = {}
        for resp in self.get_responses(look_for_commands=['list']):
            if not nodes:
                data_found[resp.get('node_id')] = resp.get('databases')
            elif resp.get('node_id') in nodes:
                data_found[resp.get('node_id')] = resp.get('databases')

        # Print dataset tables usong data_found object
        if verbose:
            for node in data_found:
                if len(data_found[node]) > 0:
                    rows = [row.values() for row in data_found[node]]
                    headers = data_found[node][0].keys()
                    info = '\n Node: {} | Number of Datasets: {} \n'.format(node, len(data_found[node]))
                    logger.info(info + tabulate.tabulate(rows, headers, tablefmt="grid") + '\n')
                else:
                    logger.info('\n Node: {} | Number of Datasets: {}'.format(node, len(data_found[node])) +
                                " No data has been set up for this node.")

        return data_found

    def training_plan_approve(self,
                              training_plan: BaseTrainingPlan,
                              description: str = "no description provided",
                              nodes: list = [],
                              timeout: int = 5) -> dict:
        """Send a training plan and a ApprovalRequest message to node(s).

        If a list of node id(s) is provided, the message will be individually sent
        to all nodes of the list.
        If the node id(s) list is None (default), the message is broadcast to all nodes.

        Args:
            training_plan: the training plan class to send to the nodes for approval.
            description: Description of training plan approval request
            nodes: list of nodes (specified by their UUID)
            timeout: maximum waiting time for the answers

        Returns:
            a dictionary of pairs (node_id: status), where status indicates to the researcher
            that the training plan has been correctly downloaded on the node side.
            Warning: status does not mean that the training plan is approved, only that it has been added
            to the "approval queue" on the node side.
        """

        # first verify all arguments
        if not isinstance(nodes, list):
            logger.error("bad nodes argument, training plan not sent")
            return {}

        if not issubclass(training_plan, BaseTrainingPlan):
            logger.error("bad training plan argument, no request sent")
            return {}

        # save and load training plan to a file to be sure
        # 1. a file is associated to training plan so we can read its source, etc.
        # 2. all dependencies are applied
        training_plan_instance = training_plan()
        training_plan_module = 'my_model_' + str(uuid.uuid4())
        with tempfile.TemporaryDirectory(dir=environ['TMP_DIR']) as tmp_dir:
            training_plan_file = os.path.join(tmp_dir, training_plan_module + '.py')
            try:
                training_plan_instance.save_code(training_plan_file)
            except Exception as e:
                logger.error(f"Cannot save the training plan to a local tmp dir : {e}")
                return {}

            try:
                _, training_plan_instance = import_class_object_from_file(
                    training_plan_file, training_plan.__name__)
                tp_source = training_plan_instance.source()
            except Exception as e:
                logger.error(f"Cannot instantiate the training plan: {e}")
                return {}

        try:
            minify(tp_source,
                   remove_annotations=False,
                   combine_imports=False,
                   remove_pass=False,
                   hoist_literals=False,
                   remove_object_base=True,
                   rename_locals=False)
        except Exception as e:
            # minify does not provide any specific exception
            logger.error(f"This file is not a python file ({e})")
            return {}

        # send message to node(s)
        message = {
            'researcher_id': environ['RESEARCHER_ID'],
            'description': str(description),
            'training_plan': tp_source,
            'command': 'approval'}

        if nodes:
            # send message to each node
            for n in nodes:
                sequence = self.send_message(message, client=n, add_sequence=True)
        else:
            # broadcast message
            sequence = self.broadcast(message, add_sequence=True)

        # wait for answers for a certain timeout
        result = {}

        for resp in self.get_responses(look_for_commands=['approval'],
                                       timeout=timeout):
            
            if sequence != resp['sequence']:
                logger.error("received an approval_reply with wrong sequence, ignoring it")
                continue

            n = resp['node_id']
            s = resp['success']
            result[n] = s

            if s:
                logger.info(f"node ({n}) has correctly downloaded the training plan")
            else:
                logger.info(f"node ({n}) has not correctly downloaded the training plan")

        # print info to the user regarding the result
        if not result or not any(result.values()):
            logger.info("no nodes have acknowledged correct training plan reception before the timeout")

        # eventually complete the result with expected results
        # (if the message was sent to specific nodes)
        for n in nodes:
            if n not in result:
                result[n] = False
                logger.info(f"node ({n}) has not acknowledge training plan reception before the timeout")

        # return the result
        return result

    def add_monitor_callback(self, callback: Callable[[Dict], None]):
        """ Adds callback function for monitor messages

        Args:
            callback: Callback function for handling monitor messages that come due 'general/monitoring' channel
        """

        self._monitor_message_callback = callback

    def remove_monitor_callback(self):
        """ Removes callback function for Monitor class. """

        self._monitor_message_callback = None
