"""gRPC Environment Servicer
"""

#####################
#                   #
#      IMPORTs      #
#                   #
#####################

from __future__ import print_function

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

# Standard library
from concurrent import futures
import importlib
import math
import os
import time

# Third party
from absl import app
from absl import flags
from absl import logging
import grpc
import gym
import mss
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends import backend_tkagg
from PIL import Image
from PIL import ImageTk
import tkinter as tk


# Local
import gym_pb2
import gym_pb2_grpc
from moog import env_wrappers
from moog import observers
from moog import environment
from moog_demos import gif_writer as gif_writer_lib
from moog_demos import human_agent
from moog.env_wrappers import gym_wrapper
from moog_demos import gui_frames
from moog import action_spaces


_WINDOW_ASPECT_RATIO = 2.7  # height/width for the gui window

FLAGS = flags.FLAGS


#####################
#                   #
#       Flags       #
#                   #
#####################

# NOTE(JP): This come from the gym_uds
# flags.DEFINE_string("id", "", "The id of the gym environment to simulate.")
flags.DEFINE_string("sockfilepath", "unix:///tmp/gym-socket", "A unique filepath where the Unix domain server will bind.")

flags.DEFINE_integer("seed", 0, "Experiment's seed.")

# NOTE(JP): This comer from Gym Wrapper
flags.DEFINE_string('config',
                    'moog_demos.example_configs.pong',
                    'Filename of task config to use.')
flags.DEFINE_integer('level', 0, 'Level of task config to run.')
flags.DEFINE_integer('render_size', 512,
                     'Height and width of the output image.')
flags.DEFINE_integer('anti_aliasing', 1, 'Renderer anti-aliasing factor.')
flags.DEFINE_integer('fps', 20,
                     'Upper bound on frames per second. Note: this is not an '
                     'accurate fps for the demo, since matplotlib and tkinter '
                     'introduce additional lag.')
flags.DEFINE_integer('reward_history', 30,
                     'Number of historical reward timesteps to plot.')

# Flags for gif writing
flags.DEFINE_boolean('write_gif', False, 'Whether to write a gif.')
flags.DEFINE_string('gif_file',
                    '/logs/gifs/test.gif',
                    'File path to write the gif to.')
flags.DEFINE_integer('gif_fps', 15, 'Frames per second for the gif.')

# Flags for logging timestep data
flags.DEFINE_boolean('log_data', False, 'Whether to log timestep data.')


#####################
#                   #
#    Environment    #
#                   #
#####################

class EnvironmentServicer(gym_pb2_grpc.EnvironmentServicer):

    def __init__(self, config):
        """This works for OpenAI type environments. MOOG is a dm_env.
        """
        self.timestep = 0
        np.random.seed(FLAGS.seed)
        self.game_name = config.split(".")[-1]
        self.experiment_id = "{}_{}".format(
            self.game_name,
            time.strftime('%Y.%m.%d_%H.%M.%S')
        )

        logging.info("[START] Init environment...")
        config = importlib.import_module(FLAGS.config)
        config = config.get_config(FLAGS.level)
        config['observers']['image'] = observers.PILRenderer(
            image_size=(FLAGS.render_size, FLAGS.render_size),
            anti_aliasing=FLAGS.anti_aliasing,
            color_to_rgb=config['observers']['image'].color_to_rgb,
            polygon_modifier=config['observers']['image'].polygon_modifier,
        )
        _env = environment.Environment(**config)
        self.env = gym_wrapper.GymWrapper(_env)

    def Reset(self, empty_request, context):
        observation = self.env.reset()
        observation_pb = gym_pb2.Observation(data=observation.ravel(), shape=observation.shape)
        return gym_pb2.State(observation=observation_pb, reward=0.0, done=False)

    def Step(self, action_request, context):
        self.timestep += 1
        logging.info("[TIMESTEP] {}".format(self.timestep))
        observation, reward, done, _ = self.env.step(action_request.value)
        self.render(observation, reward)
        observation_pb = gym_pb2.Observation(data=observation.ravel(), shape=observation.shape)
        return gym_pb2.State(observation=observation_pb, reward=reward, done=done)

    def render(self, observation, reward):
        outpath = "imgs/{}".format(self.experiment_id)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        # logging.info("[IMG] Image shape: {}".format(observation.shape))
        img = Image.fromarray(observation)
        now = str(int(time.time() * 1000))
        filename = '{}/{}.png'.format(outpath,now)
        img.save(filename)


    def Sample(self, empty_request, context):
        action = self.env.action_space.sample()
        logging.info("[ACTION] Sampled action: {}".format(action))
        # logging.info("[TYPE] Sampled action: {}".format(type(action)))
        # action = np.array([round(v) for v in action])
        # logging.info("[ROUND] Sampled action: {}".format(action))
        return gym_pb2.Action(value=action)


def remove_socket_file_path():
    try:
        os.remove(FLAGS.sockfilepath)
    except FileNotFoundError:
        pass


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    gym_pb2_grpc.add_EnvironmentServicer_to_server(EnvironmentServicer(FLAGS.config), server)
    server.add_insecure_port(FLAGS.sockfilepath)
    server.start()
    server.wait_for_termination()


#####################
#                   #
#        Main       #
#                   #
#####################

def main(_):
    remove_socket_file_path()
    serve()


if __name__ == "__main__":
    app.run(main)