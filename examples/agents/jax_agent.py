from __future__ import print_function, division, absolute_import
import argparse

import numpy as onp
from jax.scipy.special import logsumexp
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
import math
import pandas


import gym
from gym import logger, wrappers

import wandb

wandb.init()


# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


layer_sizes = [4, 100, 20, 10, 4]
# layer_sizes = [4, 8, 16, 100, 100, 6, 5]
# param_scale = 0.1


def clamp(x, bounds):
    [low, high] = bounds
    return np.minimum(np.maximum(low, x), high)


# Clamps x [-1,1]
def relu(x):
    res = np.minimum(np.maximum(-1, x), 1)
    # print("x", x)
    # print("res", res)
    return res


def derive_action(x):
    return 0.0
    # sin(x)


def predict(params, state, action_size=2, action_layer=[3, 4]):
    """
    Predict the next state give the args:
        params: network parameters
        state:  current state

    Returns:
        predicted_state: next state prediction
        action: action to take
    """

    # Date is a mutable variable that will hold the intermediatery states between layers
    data = state

    i = 0
    for w, b in params[:-1]:
        data = np.add(np.dot(w, data), b)
        i += 1

    final_w, final_b = params[-1]
    predicted_state = np.tanh(np.dot(final_w, data))

    # TODO: Make this come out of a noise function
    action = 0.0

    return predicted_state, action


# TODO: Problem: They last set of weights that feed directly to the action will
#                not update since there is no backprop. I need to make them sufficiently
#                noisey and random.
#                so that small rest of the network produce lots of noise there


def loss(params, x, y):
    predicted_state, _ = predict(params, x)
    mse = (np.square(predicted_state - y)).sum()
    return mse


# Return a vaue as the last from a jit function and we'll log it
# Make a @jit-log that logs returned values
@jit
def update(params, x, y, step_size, step):
    predicted_observation, action = predict(params, x)
    grads = grad(loss)(params, x, y)
    loss_v = loss(params, x, y)
    dws, dbs = zip(*grads)
    ws, bs = zip(*params)

    return (
        [
            (w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)
        ],
        predicted_observation,
        action,
        {
            "weights": np.concatenate([np.ravel(layer) for layer in ws]),
            "biases": np.concatenate([np.ravel(layer) for layer in bs]),
            "dweights": np.concatenate([np.ravel(layer) for layer in dws]),
            "dbiases": np.concatenate([np.ravel(layer) for layer in dbs]),
            "step_size": step_size,
            "loss": loss_v,
        },
    )

    # TODO: Make this generic for all states


def regularize(state):
    return np.array(
        [
            state[0] / 2.4,
            relu(state[1]) / 200,  # Use relu to trim outliers
            state[2] / 84,
            relu(state[3] / 300),  # Use relu to trim outliers
        ]
    )


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        self.params = init_network_params(layer_sizes, random.PRNGKey(0))
        self.parameter_history = [self.params]
        self.step_n = 1

    def parameter_history_dataframe(self):
        return pandas.DataFrame(
            data=self.parameter_history, columns=[i for i in range(0, len(self.params))]
        )

    def act(self, observation, reward, done):
        observation = regularize(observation)
        logs = {}
        params = self.params
        # First step predict and take random action
        if self.step_n == 1:
            self.predicted_state, _ = predict(params, observation)
            self.action_taken = self.action_space.sample()
        else:
            # Impact of the update is determined by learning_rate, step_size, reward
            #    * learning_rate is a fixed hyper parameter
            #    * exponential on reward
            #    * root on step_n
            learning_rate = 0.001
            magnitude = 1 / math.sqrt(self.step_n) * learning_rate * reward * reward

            # Update model
            params, predicted_observation, action, logs = update(
                params, self.predicted_state, observation, magnitude, self.step_n
            )

            self.params = params

            self.parameter_history.append(params)

            self.predicted_state = predicted_observation
            self.action_taken = round(action)

        logs["action_taken"] = self.action_taken
        logs["reward"] = reward
        logs["action"] = self.action_taken
        wandb.log(logs)

        # Increment step counte
        self.step_n = self.step_n + 1

        # return self.action_space.sample()
        return self.action_taken


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "env_id", nargs="?", default="CartPole-v0", help="Select the environment to run"
    )
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = "/tmp/random-agent-results"
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 10
    done = False

    for i in range(episode_count):
        total_reward = 0
        ob = env.reset()
        while True:
            action = agent.act(ob, total_reward, done)
            ob, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
            # Note there's no env.render() here. But the environment still can open
            # window and render if asked by env.monitor: it calls
            # env.render('rgb_array') to record video. Video is not recorded every
            # episode, see capped_cubic_video_schedule for details.

    wandb.summary.update({"parameter_history": agent.parameter_history_dataframe()})
    # Close the env and write monitor result info to disk
    env.close()
