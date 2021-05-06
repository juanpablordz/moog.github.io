"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function
import logging

import grpc

import test_pb2
import test_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = test_pb2_grpc.GreeterStub(channel)
        response = stub.SayHello(test_pb2.HelloRequest(name='you'))
        print("Greeter client received: " + response.message)
        response = stub.SayHelloAgain(test_pb2.HelloRequest(name="you"))
        print("Greeter client received: " + response.message)


if __name__ == '__main__':
    logging.basicConfig()
    run()
