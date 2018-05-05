# COGS-298 Project: Understanding Reinforcement Learning Through OpenAI Gym and Pong
## Overview
This project is an attempt at both gaining a better understanding of the concepts behind reinforcement learning (RL) and imparting such knowledge onto others, by taking a closer look at Andrej Karpathy's RL [tutorial](#sources). Karpathy's work provides an excellent example of implementing and explaining complex features of RL algorithms like policy gradients, a method of improving the policy, an arbitrary rule used to determine what the agent should do based on the current state of the environment [(Sutton and Barto, 2018)](#sources). His post however, doesn't detail much of the general information about RL needed to fully understand what is happening. Explicitly explaining the components of RL should make the process of developing algorithms easier by providing a basis of considerations that must be implemented. Teaching an agent to play Pong is a deceivingly challenging task that will be examined closer.

## Background

### Markov Decision Processes
Reinforcement learning is a field of machine learning focused on agent-environment interaction. The objective of problems that utilize RL methods is to find a policy, or the actions of an agent, that yield the highest total reward over the lifespan of training. In the context of RL, the agent is the decision maker. It receives observations about the current state of the environment and takes an action based on that information. The agent then receives a reward based on the chosen action. This reward, along with a new state, is passed to the agent and another action is taken. The environment includes whatever the agent has no control over. This feedback loop is known as a Markov Decision Process (MDP) [(Sutton and Barto, 2018)](#sources). 


<p align = center>
  <img src = figures/mdp.png>
  </p>
  
*Fig.1: A simplified diagram of the MDP. At each time step, the state of the environment is presented to the agent, who then makes a decision. The state of the environment changes and the agent is given a reward; a positive, negative, or neutral value based on the action taken. These observations are given to the agent to make another decision with (Sutton and Barto, 2018).*


The reward is calculated as a function of the current state of the environment and the action taken [(Alzantot, 2017)](#sources).  In the case of Pong, if the ball goes past the agent, the agent receives a reward of -1. If the agent hits the ball past the opponent, the reward given is +1. For all other state/action pairs, the environment returns no reward [(Karpathy, 2016)](#sources). In a complete MDP model, all of the information needed to calculate the policy of the agent is be available. Besides states, actions, and rewards (along with a [discount factor](#discounted-reward)), the MDP model requires a state transition probability, the likelihood of moving from one state to another given the action taken. With this information, the optimal policy can be calcuated directly through techniques like policy iteration. With policy iteration, the policy of each state is continually updated until the expected future reward for each state is maximized [(Kaelbling, 1996)](#sources). Unfortunately, this information is not always explictly given. In the case of Pong, the agent is given no innate knowledge of how the game operates from one pixel frame to the next. It must learn such probabilities from playing the game. The optimal policy can be calculated through trial and error, turning the model into one of RL rather than strictly MDP.

### MDP Challenges
When creating a MDP framework, there are additional considerations that must be made. It will help to describe some of issues that arise when applying the MDP approach, since some of the problems are not obvious.
#### Discounted Reward
As mentioned when discussing the structure of MDP problems, one of the parameters of the model is the discount factor, typically denoted as gamma (γ). Below is the mathematical formula for calculating the discounted return.


<p align = center>
  <img src = figures/return-formula.png>
  </p>
  
*Fig.2: Discounted return formula [Sutton and Barto, 2018](#sources). The return is the sum of the reward at each time step. A discount factor is multiplied exponentially to each reward to lessen the importance of future rewards*


Gamma is a value between 0 and 1, exclusive. The discount factor serves several purposes. It allows the total return to converge at a finite value since the reward grows exponentially smaller over time [Silver, 2017](#sources).  Having a gamma value close to 1 makes the agent more 'far-sighted', since the value of later rewards are important in maximizing the total reward earned. In such a case, the agent needs to consider how future states and actions will yield rewards, not just the current one. A gamma value near 0 de-emphasizes future rewards. Setting a low discount factor will help maximize the current reward but can lower the overall return since decisions aren't being made with the future in mind [Sutton and Barto, 2018](#sources). Besides the mathematical benefit, discount factors can also highlight the uncertainty of the environment. The agent should treat future rewards (state/action pairs returning a positive or negative value) with less importance when finding a policy since there is no guarantee of a future state. This will be expaned upon when describing policy gradients in more detail.

#### Choosing Goals
Another important piece of the MDP problem is deciding on 
#### Exploration vs. Exploitation
...


### Reinforcement Learning
#### Policy Gradients


### OpenAI Gym

### Code
http://minpy.readthedocs.io/en/latest/tutorial/rl_policy_gradient_tutorial/rl_policy_gradient.html (explains code a bit)
#### Notes (to be deleted)


How we get highest reward: optimize the policy -- PI
How to we do that? Policy gradients
Policy search refers to methods that directly learn the policy for solving a Markov Decision Process (MDP) and Policy gradients are a subset of this wide class of algorithms.

Policy gradients -- no value function -- directly change policy -- create a policy 




## Sources
[1] Karpathy, A. (2016, May 31). Deep Reinforcement Learning: Pong from Pixels. Retrieved from http://karpathy.github.io/2016/05/31/rl/

[2] Sutton & Barto

[3] https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

[4] https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node20.html

[5] Open AI Gym

[6] http://www.cs.ubc.ca/~murphyk/Bayes/pomdp.html

[7] http://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=1539&context=etd_projects
