# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides contextmanager which manages a gRPC channel to connect to the
server."""
from contextlib import contextmanager
from logging import DEBUG
from queue import Queue
from typing import Callable, Iterator, Tuple

import grpc

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.proto.transport_pb2 import ClientMessage, ServerMessage
from flwr.proto.transport_pb2_grpc import FlowerServiceStub

# Uncomment these flags in case you are debugging
# os.environ["GRPC_VERBOSITY"] = "debug"
# os.environ["GRPC_TRACE"] = "connectivity_state"


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)


# Context manager:
# https://book.pythontips.com/en/latest/context_managers.html
# https://docs.python.org/3/library/contextlib.html#contextlib.contextmanager
@contextmanager
def insecure_grpc_connection(
    server_address: str, max_message_length: int = GRPC_MAX_MESSAGE_LENGTH
) -> Iterator[Tuple[Callable[[], ServerMessage], Callable[[ClientMessage], None]]]:
    """Establish an insecure gRPC connection to a gRPC server."""
    # https://grpc.github.io/grpc/python/grpc.html#create-client
    channel = grpc.insecure_channel(
        server_address,
        options=[
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
        ],
    )
    # https://grpc.github.io/grpc/python/grpc.html#grpc.Channel.subscribe
    channel.subscribe(on_channel_state_change)

    queue: Queue[ClientMessage] = Queue(  # pylint: disable=unsubscriptable-object
        maxsize=1
    )
    stub = FlowerServiceStub(channel)  # type: ignore

    # iter(Callable, None)
    # https://stackoverflow.com/questions/38087427/what-are-the-uses-of-itercallable-sentinel
    # https://amir.rachum.com/blog/2013/11/10/python-tips-iterate-with-a-sentinel-value/
    # rpc Join(stream ClientMessage) returns (stream ServerMessage) {}
    # 
    # On the client side, the client has a local object known as stub (for some 
    # languages, the preferred term is client) that implements the same methods 
    # as the service. The client can then just call those methods on the local 
    # object, wrapping the parameters for the call in the appropriate protocol 
    # buffer message type - gRPC looks after sending the request(s) to the 
    # server and returning the server’s protocol buffer response(s).
    server_message_iterator: Iterator[ServerMessage] = stub.Join(iter(queue.get, None))

    receive: Callable[[], ServerMessage] = lambda: next(server_message_iterator)
    send: Callable[[ClientMessage], None] = lambda msg: queue.put(msg, block=False)

    try:
        yield (receive, send)
    finally:
        # Make sure to have a final
        channel.close()
        log(DEBUG, "Insecure gRPC channel closed")
