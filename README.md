# COGS-298 Project: Understanding Reinforcement Learning Through OpenAI Gym and Pong
## Background
This project is an attempt at both gaining a better understanding of the concepts behind reinforcement learning (RL) and imparting such knowledge onto others, by taking a closer look at Andrej Karpathy's RL [tutorial](#sources). Karpathy's work provides an excellent example of implementing and explaining complex features of RL algorithms like policy gradients, a method of improving the policy, an arbitrary rule used to determine what the agent should do based on the current state of the environment [(Sutton and Barto, 2018)](#sources). His post however, doesn't detail much of the general information about RL needed to fully understand what is happening. Explicitly explaining the components of RL should make the process of developing algorithms easier by providing a basis of considerations that must be implemented. Teaching an agent to play Pong is a deceivingly challenging task that will be examined closer.

## Reinforcement learning
Reinforcement learning is a field of machine learning focused on agent-environment interaction. The objective of problems that utilize RL methods is to find a policy, or the actions of an agent, that yields the highest total reward over the lifespan of training. In the context of RL, the agent is the decision maker, it receives observations about the current state of the environment and takes an action based on that information. The agent then receives a reward based on the chosen action. This reward, along with a new state, is passed to the agent and another action is taken. This feedback loop is known as a Markov Decision Process (MDP) [(Sutton and Barto, 2018)](#sources). The environment includes whatever the agent has no control over. 


<p align = center>
  <img src = figures/mdp.png>
  </p>
  
*Fig.1: A simplified diagram of of the MDP. At each time step, the state of the environment is presented to the agent, who then makes a decision. The environment responds by returning a reward; a positive, negative, or neutral value based on the action taken. This information is given to the agent to make another decision with (Sutton and Barto, 2018). This description glosses over a key part of the MDP, how exactly states, actions, and rewards interact with eachother. This interaction is described below.*










## Sources
[1] Karpathy, A. (2016, May 31). Deep Reinforcement Learning: Pong from Pixels. Retrieved from http://karpathy.github.io/2016/05/31/rl/

[2] Sutton & Barto

[3] https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

[4] Open AI Gym

[5] http://www.cs.ubc.ca/~murphyk/Bayes/pomdp.html
