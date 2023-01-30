# AI-using-games

import random
import numpy as np
import gym

# Define the environment for the game
env = gym.make("CartPole-v1")

# Define the AI's policy
def policy(observation, theta):
    angle = observation[2]
    if angle < 0:
        action = 0
    else:
        action = 1
    return action

# Define the function that calculates the discounted reward
def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros_like(rewards)
    cumulative_rewards = 0
    for i in reversed(range(len(rewards))):
        cumulative_rewards = rewards[i] + cumulative_rewards * discount_rate
        discounted_rewards[i] = cumulative_rewards
    return discounted_rewards

# Train the AI
n_episodes = 1000
discount_rate = 0.95
theta = np.zeros(4)

for episode in range(n_episodes):
    observation = env.reset()
    episode_rewards = []
    for t in range(100):
        action = policy(observation, theta)
        next_observation, reward, done, info = env.step(action)
        episode_rewards.append(reward)
        if done:
            break
        observation = next_observation
    discounted_rewards = discount_rewards(episode_rewards, discount_rate)
    print("Episode ", episode, ": reward = ", sum(episode_rewards), " discounted reward = ", np.sum(discounted_rewards))

# Test the AI
observation = env.reset()
for t in range(100):
    env.render()
    action = policy(observation, theta)
    observation, reward, done, info = env.step(action)
    if done:
        break

env.close()
