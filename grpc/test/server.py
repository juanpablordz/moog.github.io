"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging

import grpc

import test_pb2
import test_pb2_grpc


class Greeter(test_pb2_grpc.GreeterServicer):

    def SayHello(self, request, context):
        return test_pb2.HelloReply(message='Hello, %s!' % request.name)

    def SayHelloAgain(self, request, context):
        return test_pb2.HelloReply(message="Hello again, {}".format(request.name))


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    test_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
