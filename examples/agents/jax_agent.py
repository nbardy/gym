from __future__ import print_function, division, absolute_import
import argparse

import numpy as onp
from jax.scipy.special import logsumexp
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random

import math

import gym
from gym import logger, wrappers


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


layer_sizes = [5, 8, 16, 8, 5]
param_scale = 0.1
step_size = 0.0001
num_epochs = 10
batch_size = 128
n_targets = 10


def relu(x):
    return np.minimum(np.maximum(0, x), 1)


def predict(params, state):
    # per-example predictions
    activations = state
    for w, b in params[:-1]:
        outputs = np.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits, action = np.split(np.dot(final_w, activations), [-1])
    return logits, np.amin(relu(action).astype(np.int32))


# Make a batched version of the `predict` function
# batched_predict = vmap(predict, in_axes=(None, 0))


def loss(params, x, y):
    mse = (np.square(x - y)).mean()
    return mse


@jit
def update(params, x, y, reward):

    step_size = (10 + reward) / 10

    grads = grad(loss)(params, x, y)
    return [
        (w - step_size * dw, b - step_size * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        self.params = init_network_params(layer_sizes, random.PRNGKey(0))

    def act(self, observation, reward, done):
        # Update model, besdies first go
        if hasattr(self, "predicted_state"):
            update(self.params, self.predicted_state, observation, reward)

        input = np.concatenate((observation, np.array([reward])))
        predicted_observation, action = predict(self.params, input)

        self.predicted_state = predicted_observation
        self.action_taken = action
        return self.action_space.sample()
        # return self.action_taken


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

    episode_count = 100
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
