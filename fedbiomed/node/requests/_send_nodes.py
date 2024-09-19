# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Tuple

from fedbiomed.common.constants import TIMEOUT_NODE_TO_NODE_REQUEST
from fedbiomed.common.message import InnerMessage, NodeMessages, InnerRequestReply
from fedbiomed.common.synchro import EventWaitExchange

from fedbiomed.transport.controller import GrpcController

from fedbiomed.node.environ import environ
from ._n2n_router import NodeToNodeRouter


def send_nodes(
        n2n_router: NodeToNodeRouter,
        grpc_client: GrpcController,
        pending_requests: EventWaitExchange,
        researcher_id: str,
        nodes: List[str],
        messages: List[InnerMessage],
) -> Tuple[bool, List[Any]]:
    """Send message to some other nodes using overlay communications and wait for their replies.
        Args:
            n2n_router: object managing node to node messages
            grpc_client: object managing the communication with other components
            pending_requests: object for receiving overlay node to node reply message
            researcher_id: unique ID of researcher connecting the nodes
            nodes: list of node IDs of the destination nodes
            messages: list of the inner messages for the destination nodes
        Returns:
            status: True if all messages are received
            replies: List of replies from each node.
    """
    request_ids = []

    for node, message in zip(nodes, messages):
        overlay, salt = n2n_router.format_outgoing_overlay(message, researcher_id)
        message_overlay = NodeMessages.format_outgoing_message(
            {
                'researcher_id': researcher_id,
                'node_id': environ['NODE_ID'],
                'dest_node_id': node,
                'overlay': overlay,
                'setup': False,
                'salt': salt,
                'command': 'overlay',
            })

        grpc_client.send(message_overlay)

        if isinstance(message, InnerRequestReply):
            request_ids += [message.get_param('request_id')]

    ret =  pending_requests.wait(request_ids, TIMEOUT_NODE_TO_NODE_REQUEST)
    return ret