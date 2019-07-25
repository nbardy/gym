from __future__ import print_function, division, absolute_import
import argparse

import numpy as onp
from jax.scipy.special import logsumexp
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random


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


layer_sizes = [4, 8, 16, 5]
layer_sizes = [4, 8, 16, 100, 100, 6, 5]
param_scale = 0.1


def relu(x):
    return np.minimum(np.maximum(0, x), 1)


def predict(params, state):
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
    for w, b in params[:-1]:
        data = np.add(np.dot(w, data), b)
        data = relu(data)

    final_w, final_b = params[-1]
    predicted_state, action = np.split(relu(np.dot(final_w, data)), [-1])
    return predicted_state, action


def loss(params, x, y):
    predicted_state, _ = predict(params, x)
    mse = (np.square(predicted_state - y)).sum()
    return mse


# Return a vaue as the last from a jit function and we'll log it
# Make a @jit-log that logs returned values
@jit
def update(params, x, y, reward, step):
    step_size = (10 + reward) / 10000 * (1 / step * step)
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


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        self.params = init_network_params(layer_sizes, random.PRNGKey(0))
        self.step_n = 1

    def act(self, observation, reward, done):
        logs = {}
        params = self.params
        # First step predict and take random action
        if self.step_n == 1:
            self.predicted_state, _ = predict(params, observation)
            self.action_taken = self.action_space.sample()
        else:
            # Update model
            params, predicted_observation, action, logs = update(
                params, self.predicted_state, observation, reward, self.step_n
            )
            action = float(action[0])
            logs["action"] = action
            self.params = params

            self.predicted_state = predicted_observation
            self.action_taken = round(action)

        logs["action_taken"] = self.action_taken
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
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open
            # window and render if asked by env.monitor: it calls
            # env.render('rgb_array') to record video. Video is not recorded every
            # episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
