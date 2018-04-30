# COGS-298 Project: Understanding Reinforcement Learning Through OpenAI Gym and Pong
## Background
This project is an attempt at both gaining a better understanding of the concepts behind reinforcement learning (RL) and imparting such knowledge onto others, by taking a closer look at Andrej Karpathy's RL [tutorial](#sources). Karpathy's work provides an excellent example of implementing and explaining complex features of RL algorithms like policy gradients, a method of improving the policy, an arbitrary rule used to determine what the agent should do based on the current state of the environment [(Sutton and Barto, 2018)](#sources). His post however, doesn't detail much of the general information about RL needed to fully understand what is happening. Explicitly explaining the components of RL should make the process of developing algorithms easier by providing a basis of considerations that must be implemented. Teaching an agent to play Pong is a deceivingly challenging task that will be examined closer.

## Reinforcement learning

### Markov Decision Processes
Reinforcement learning is a field of machine learning focused on agent-environment interaction. The objective of problems that utilize RL methods is to find a policy, or the actions of an agent, that yield the highest total reward over the lifespan of training. In the context of RL, the agent is the decision maker. It receives observations about the current state of the environment and takes an action based on that information. The agent then receives a reward based on the chosen action. This reward, along with a new state, is passed to the agent and another action is taken. The environment includes whatever the agent has no control over. This feedback loop is known as a Markov Decision Process (MDP) [(Sutton and Barto, 2018)](#sources). 


<p align = center>
  <img src = figures/mdp.png>
  </p>
  
*Fig.1: A simplified diagram of of the MDP. At each time step, the state of the environment is presented to the agent, who then makes a decision. The state of the environment changes and the agent is given a reward; a positive, negative, or neutral value based on the action taken. These observations are given to the agent to make another decision with (Sutton and Barto, 2018).*




The reward is calculated as a function of the current state of the environment and the action taken [(Alzantot, 2017)](#sources).  In the case of Pong, if the ball goes past the agent, the agent receives a reward of -1. If the agent hits the ball past the opponent, the reward given is +1. For all other state/action pairs, the environment returns a reward of 0 [(Karpathy, 2016)](#sources). In a normal 


The state, action, and reward provide 
*Talk about what else is needed in a Markov Decision Process - state transition probability (we don't have), that is needed to find the policy of the 

### MDP Problems
When creating a MDP framework, there are additional considerations that must be made. It will help to describe some of issues that arise when applying the MDP approach, since some of the problems are not obvious.
#### Discounted Reward
When calculating the total return


This description glosses over a key part of the MDP structure: how  states, actions, and rewards interact with eachother. As previously mentioned, the goal of RL is to maximize the return, the total reward earned. 
#### Notes (to be deleted)
must include a few things: discount factor (explain why), state transition probability, reward function -- how they interact with eachother?
Reward function R(s,a) based on the state and the action taken, can either return -1,0,or 1
Discount factor gamma - between 0 and 1 -- explain why we need to discount
Probability of state transition

How we get highest reward: optimize the policy -- PI
How to we do that? Policy gradients
Policy search refers to methods that directly learn the policy for solving a Markov Decision Process (MDP) and Policy gradients are a subset of this wide class of algorithms.

Policy gradients -- no value function -- directly change policy




## Sources
[1] Karpathy, A. (2016, May 31). Deep Reinforcement Learning: Pong from Pixels. Retrieved from http://karpathy.github.io/2016/05/31/rl/

[2] Sutton & Barto

[3] https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

[4] Open AI Gym

[5] http://www.cs.ubc.ca/~murphyk/Bayes/pomdp.html
