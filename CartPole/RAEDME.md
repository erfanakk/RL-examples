
# Cartpole-RL-DQN

<br>
<br>

## Cartpole Problem
Cartpole - known also as an Inverted Pendulum is a pendulum with a center of gravity above its pivot point. It’s unstable, but can be controlled by moving
the pivot point under the center of mass. The goal is to keep the cartpole balanced by applying appropriate forces to a pivot point.

<br>

<p align="center">
 <img src = "cartpole.png">
</p> 

1. Violet square indicates a pivot point
2. Red and green arrows show possible horizontal forces that can be applied to a pivot point

<p>A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the
cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The
episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.</p>

<br>

## Reinforcement Learning
In order to achieve the desired behavior of an agent that learns from its mistakes and improves its performance, we need to get more familiar with the concept of
Reinforcement Learning (RL).

RL is a general concept that can be simply described with an agent that takes actions in an environment in order to maximize its cumulative reward. The underlying
idea is very lifelike, where similarly to the humans in real life, agents in RL algorithms are incentivized with punishments for bad actions and rewards for good
ones.

<br>

## OpenAI Gym
This project is based on top of OpenAI’s gym and for those of you who are not familiar with the gym - I’ll briefly explain it.
Long story short, gym is a collection of environments to develop and test RL algorithms. Cartpole is one of the available gyms, you can check the full list 
[here](https://gym.openai.com/envs/#classic_control).

It’s built on a Markov chain model that is illustrated below.

<br>

<p align="center">
 <img src = "Markov_chain.png">
</p> 

We start with an initial environment. It doesn’t have any associated reward yet, but it has a state (S_t).

Then for each iteration, an agent takes current state (S_t), picks best (based on model prediction) action (A_t) and executes it on an environment. Subsequently,
environment returns a reward (R_t+1) for a given action, a new state (S_t+1) and an information if the new state is terminal. The process repeats until 
termination.

<br>

## Deep Q-Learning (DQN)
DQN is a RL technique that is aimed at choosing the best action for given circumstances (observation). Each possible action for each possible observation has its
Q value, where ‘Q’ stands for a quality of a given move.

Here is a formal notation for updating our Q-value:

<br>

<p align="center">
 <img src = "Q_learning.png">
</p> 

<br>
