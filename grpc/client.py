"""gRPC Client.
"""

#####################
#                   #
#      IMPORTs      #
#                   #
#####################

from __future__ import print_function

import sys
sys.path.insert(0, '..')

# Standard library
import os

# Third party
from absl import app
from absl import flags
from absl import logging
import grpc
import numpy as np

# Local
import gym_pb2
import gym_pb2_grpc

FLAGS = flags.FLAGS

#####################
#                   #
#       Flags       #
#                   #
#####################

# NOTE(JP): This come from the gym_uds
# flags.DEFINE_string("id", "", "The id of the gym environment to simulate.")
flags.DEFINE_string("sockfilepath", "unix:///tmp/gym-socket", "A unique filepath where the Unix domain server will bind.")

class EnvironmentClient:
    def __init__(self, sockfilepath):
        channel = grpc.insecure_channel(sockfilepath)
        self.stub = gym_pb2_grpc.EnvironmentStub(channel)
        self.action_space = lambda: None
        self.action_space.sample = self.sample # TODO: dudoso

    def reset(self):
        state_pb = self.stub.Reset(gym_pb2.Empty())
        observation = np.asarray(state_pb.observation.data).reshape(state_pb.observation.shape)
        return observation

    def step(self, action):
        state_pb = self.stub.Step(gym_pb2.Action(value=action))
        observation = np.asarray(state_pb.observation.data).reshape(state_pb.observation.shape)
        return observation, state_pb.reward, state_pb.done

    def sample(self):
        action_pb = self.stub.Sample(gym_pb2.Empty())
        return action_pb.value

def run():
    logging.info("[START] Create environment...")
    env = EnvironmentClient(FLAGS.sockfilepath)

    num_episodes = 1
    for episode in range(1, num_episodes + 1):
        logging.info("[EPSISODE] Run episode {}".format(episode))
        observation = env.reset()
        # NOTE(JP): Aqui vamos.
        episode_reward = 0
        done = False
        count = 0
        while not done:
            action = env.action_space.sample() # BUG: Do something else. WE probably need to do something of the sort access the action space of the environment.
            logging.info("[ACTION-{}] action: {}".format(count, action))
            observation, reward, done = env.step(action)
            logging.info("[STEP {}] Action:{} - Reward:{} - Done:{}".format(
                count, action, reward, done
            ))
            count += 1
            episode_reward += reward
        logging.info("Ep. {}: {:.2f}".format(episode, episode_reward))


#####################
#                   #
#        Main       #
#                   #
#####################

def main(_):
    run()


if __name__ == "__main__":
    app.run(main)