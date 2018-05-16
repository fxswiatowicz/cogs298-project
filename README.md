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


The reward is calculated as a function of the current state of the environment and the action taken [(Alzantot, 2017)](#sources).  In Pong, if the ball goes past the agent, the agent receives a reward of -1. If the agent hits the ball past the opponent, the reward given is +1. For all other state/action pairs, the environment returns no reward [(Karpathy, 2016)](#sources). In a complete MDP model, all of the information needed to calculate the policy of the agent is be available. Besides states, actions, and rewards (along with a [discount factor](#discounted-reward)), the MDP model requires a state transition probability, the likelihood of moving from one state to another given the action taken. With this information, the optimal policy can be calcuated directly through techniques like policy iteration. With policy iteration, the policy of each state is continually updated until the expected future reward for each state is maximized [(Kaelbling, 1996)](#sources). Unfortunately, this information is not always explictly given. In the case of Pong, the agent is given no innate knowledge of how the game operates from one pixel frame to the next. It must learn such probabilities from playing the game. The optimal policy can be calculated through trial and error, turning the model into one of RL rather than strictly MDP.

### MDP Challenges
When creating a MDP framework, there are additional considerations that must be made. It will help to describe a couple issues that can arise when applying the MDP approach, since some of the problems are not obvious.
#### Discounted Reward
As mentioned when discussing the structure of MDP problems, one of the parameters of the model is the discount factor, typically denoted as gamma (γ). Below is the mathematical formula for calculating the discounted return.


<p align = center>
  <img src = figures/return-formula.png>
  </p>
  
*Fig.2: Discounted return formula [(Sutton and Barto, 2018)](#sources). The return is the sum of the reward at each time step. A discount factor is multiplied exponentially to each reward to lessen the importance of future rewards*


Gamma is a value between 0 and 1, exclusive. The discount factor serves several purposes. It allows the total return to converge at a finite value since the reward grows exponentially smaller over time [(Silver, 2017)](#sources).  Having a gamma value close to 1 makes the agent more 'far-sighted', since the value of later rewards are important in maximizing the total reward earned. In such a case, the agent needs to consider how future states and actions will yield rewards, not just the current one. A gamma value near 0 de-emphasizes future rewards. Setting a low discount factor will help maximize the current reward but can lower the overall return since decisions aren't being made with the future in mind [(Sutton and Barto, 2018)](#sources). Besides the mathematical benefit, discount factors can also highlight the uncertainty of the environment. The agent should treat future rewards (state/action pairs returning a positive or negative value) with less importance when finding a policy since there is no guarantee of a future state.

#### Choosing Goals
Another important piece of the MDP problem is deciding on what goals the agent must achieve [(Sutton and Barto, 2018)](#sources). For Pong, it is clear that the agent should be rewarded positively for scoring a point and be rewarded negatively for conceding a point. Should the agent also be rewarded reaching a subgoal, like making contact with the ball without regard to where the ball goes afterward? The answer is no. The gif below is a visual example of what could happen if subgoals were rewarded. 

<p align = center>
  <img src = figures/pong-rally.gif>
  </p>
  
*Fig.3: Example of human versing AI opponent in Pong with the goal of making contact with the ball. While not allowing the ball past one's paddle is an essential part of winning Pong, it doesn't directly result in victory. To score, the ball must be paddled back to the opponent at such an angle and velocity that makes it difficult for the ball to be served back.*

Rewarding behaviors not directly associated with completing the objective of the environment can lead to policies that maximize the total return but fail to accomplish the overall goal of the game. In Pong, if the agent were rewarded every time it hit the ball, regardless of whether or not the ball goes past the opponent AI, the agent could potentially maximize the total reward earned by rallying with the opponent, hitting the ball back and forth perpetually. The agent places priority on scoring as much as possible instead of winning the game [(Clark and Amodei, 2016)](#sources). A good way to avoid this situation is to only reward 'direct' goals, actions that explicitly produce a result leading to victory or failure. For the sake of most RL problems, it doesn't matter how the model calculates the optimal policy, just that it completes the task at hand [(Sutton and Barto, 2018)](#sources).

## OpenAI Gym
OpenAI Gym provides an interface to access RL environments. Instead of having to hard-code features like rewards and possible actions to be taken by the agent, Gym allows access to the environments through the use of pre-defined methods [(Brockman et al., 2016)](#sources). These methods make it simpler to set up algorithms for solving parts of an MDP. There are several functions important to understanding how the code works, described below.

- ```env_reset()```: When called, this method resets the environment. This is necessary for the *trial and error* nature of RL since the agent should be provided a clean slate to act in after each discrete time step. In the case of Pong, the game should be reset after a game is finished (either the agent or opponent scores 21 points).

- ```env.step()```: This method performs an action in the environment. It is given a single parameter, the action the agent should take. This value is usually calculated from a probability distribution [(Karparthy, 2016)](#sources). When called, env_step() returns four values, three of which are critical for traversing through the RL algorithm: the state of the environment, the reward earned from performing the action, and whether another action should be performed (dependent on if the agent has won or lost).

- ```env.render()```: When set to true, this function will display a visualization of the agent performing actions in the environment. This can be useful as it provides concrete evidence of the agent's learning rather than relying soley on numerical values returned by calling methods like env_step().


## Code
With most of the prerequisite information regarding MDPs and RL algorithms explained, a real example of agent-environment interaction can be considered in the form of Pong. Specifcally, a closer look at Karpathy's RL code will be taken. The algorithm makes use of machine learning techniques to continually update the policy of the agent from episode to episode.

### Relationship to Research
Reinforcement learning models lean on prior psychological research on how sensory inputs turn into actions in animals [(Mnih et al., 2015)](#sources). In the case of a RL model, the sensory information given to an agent is in the form of pixel values corresponding to an on-screen game state. With this data, the agent must, through trial and error, learn how choose actions that best maximize the total reward earned from intercting with the environment.

### Input Data
In order to maximize the total reward, the agent must optimize the actions it takes in its environment (the policy). Following the MDP diagram, the agent must have access to the states and rewards given. The states are provided as an input layer of pixel information. This info is propagated through the network and provides an output that is the probability of going up or down. A snippet of code below shows this in action.

``` python
#init model
D = 80 * 80 #input dimensionality
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    #initializing rates pseudo-randomly (Xavier initialization)
    #Xavier initialization: taking the hidden nodes into account
    #when we intialize nodes (http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)
    #W1: input D computed into some vector
    #W2: dealing only with hidden weights
    model['W1'] = np.random.randn(H,D) / np.sqrt(D)
    model['W2'] = np.random.randn(H) / np.sqrt(H)
#gradient buffer helps with backpropagation. Used to store gradients.
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } 
## rmsprop (gradient descent) memory used to update model
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } 

#activation function
#sigmoid is used at end of backpropagation to return values as probabilities
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) #squashing(converting vectors into probabilities)
    
#forward propgation:
def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0 #ReLU: take the max between 0 and h
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h #return propability of taking action 2 and hidstate
```

### Output Data

### Model Design




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

[7] David Silver's lecture notes week 2

[7] http://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=1539&context=etd_projects

Clark and Amodei, 2016 - https://blog.openai.com/faulty-reward-functions/

Mnih et al., 2015 https://www.nature.com/articles/nature14236
