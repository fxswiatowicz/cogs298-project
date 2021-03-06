# COGS-298 Project: Understanding Reinforcement Learning Through Markov Decision Processes and Pong
## Overview
This project is an attempt at both gaining a better understanding of the concepts behind reinforcement learning (RL) and imparting such knowledge onto others, by taking a closer look at the background behind Andrej Karpathy's RL [tutorial](#sources). Karpathy's work provides an excellent example of implementing and explaining complex features of RL algorithms like policy gradients, a method of improving the policy, an arbitrary rule used to determine what the agent should do based on the current state of the environment [(Sutton and Barto, 2018)](#sources). His post, however, doesn't detail much of the general information about RL needed to fully understand what is happening. Explicitly explaining the components of RL should make the process of developing algorithms easier by providing a basis of considerations that must be implemented. Teaching an agent to play Pong is a deceivingly challenging task that will be examined closer.

## Background

### Markov Decision Processes
Reinforcement learning is a field of machine learning focused on agent-environment interaction. The objective of problems that utilize RL methods is to find a policy, or the actions of an agent, that yield the highest total reward over the lifespan of training. In the context of RL, the agent is the decision maker. It receives observations about the current state of the environment and takes an action based on that information. The agent then receives a reward based on the chosen action. This reward, along with a new state, is passed to the agent and another action is taken. The environment includes whatever the agent has no control over. This feedback loop is known as a Markov Decision Process (MDP) [(Sutton and Barto, 2018)](#sources). 


<p align = center>
  <img src = figures/mdp.png>
  </p>
  
*Fig.1: A simplified diagram of the MDP. At each time step, the state of the environment is presented to the agent, who then makes a decision. The state of the environment changes and the agent is given a reward; a positive, negative, or neutral value based on the action taken. These observations are given to the agent to make another decision with (Sutton and Barto, 2018).*  


The reward is calculated as a function of the current state of the environment and the action taken [(Alzantot, 2017)](#sources).  In Pong, if the ball goes past the agent, the agent receives a reward of -1. If the agent hits the ball past the opponent, the reward given is +1. For all other state/action pairs, the environment returns no reward [(Karpathy, 2016)](#sources). In a complete MDP model, all of the information needed to calculate the policy of the agent is be available. Besides states, actions, and rewards (along with a [discount factor](#discounted-reward)), the MDP model requires a state transition probability, the likelihood of moving from one state to another given the action taken. With this information, the optimal policy can be calculated directly through techniques like policy iteration. With policy iteration, the policy of each state is continually updated until the expected future reward for each state is maximized [(Kaelbling, 1996)](#sources). Unfortunately, this information is not always expliictly given. In the case of Pong, the agent is given no innate knowledge of how the game operates from one pixel frame to the next. It must learn such probabilities from playing the game. The optimal policy can be calculated through trial and error, turning the model into one of RL rather than strictly MDP.

### MDP Challenges
When creating an MDP framework, there are additional considerations that must be made. It will help to describe a couple issues that can arise when applying the MDP approach since some of the problems are not obvious.
#### Discounted Reward
As mentioned when discussing the structure of MDP problems, one of the parameters of the model is the discount factor, typically denoted as gamma (γ). Below is the mathematical formula for calculating the discounted return.


<p align = center>
  <img src = figures/return-formula.png>
  </p>
  
*Fig.2: Discounted return formula [(Sutton and Barto, 2018)](#sources). The return is the sum of the reward at each time step. A discount factor is multiplied exponentially to each reward to lessen the importance of future rewards*


Gamma is a value between 0 and 1, exclusive. The discount factor serves several purposes. It allows the total return to converge at a finite value since the reward grows exponentially smaller over time [(Silver, 2017)](#sources).  Having a gamma value close to 1 makes the agent more 'far-sighted' since the value of later rewards are important in maximizing the total reward earned. In such a case, the agent needs to consider how future states and actions will yield rewards, not just the current one. A gamma value near 0 de-emphasizes future rewards. Setting a low discount factor will help maximize the current reward but can lower the overall return since decisions aren't being made with the future in mind [(Sutton and Barto, 2018)](#sources). Besides the mathematical benefit, discount factors can also highlight the uncertainty of the environment. The agent should treat future rewards (state/action pairs returning a positive or negative value) with less importance when finding a policy since there is no guarantee of a future state.

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

- ```env.render()```: When set to true, this function will display a visualization of the agent performing actions in the environment. This can be useful as it provides concrete evidence of the agent's learning rather than relying solely on numerical values returned by calling methods like env_step().


## Code
With most of the prerequisite information regarding MDPs and RL algorithms explained, a real example of agent-environment interaction can be considered in the form of Pong. Specififcally, a closer look at Karpathy's RL code will be taken. The algorithm makes use of machine learning techniques to continually update the policy of the agent from episode to episode.

### Relationship to Research
Reinforcement learning models lean on prior psychological research on how sensory inputs turn into actions in animals [(Mnih et al., 2015)](#sources). In the case of an RL model, the sensory information given to an agent is in the form of pixel values corresponding to an on-screen game state. With this data, the agent must, through trial and error, learn how to choose actions that best maximize the total reward earned from interacting with the environment. Pong is the perfect environment for this due to well-defined states (pixel values from the game), rewards (win/lose), and actions (move up/down).

### Model Design
In order to maximize the total reward, the agent must optimize the actions it takes in its environment (the policy). Following the MDP diagram, the agent must have access to the states and rewards given. The states are provided as an input layer of pixel information. This info is propagated through the network and provides an output that is the probability of going up or down. The code below shows this in action.

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

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0
        #increment sum
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r
    
#forward propgation:
def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0 #ReLU: take the max between 0 and h
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h #return probability of taking action 2 and hidstate
   
   
# forward the policy network and sample an action from the returned probability
aprob, h = policy_forward(x)
action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
```

In this snippet of code, the model is initialized. ```W1``` and ```W2``` make up the policy network [(Karpathy, 2016)](#sources). The policy network is initialized with random values initially. During forward propagation, ```policy_forward()``` takes in ```x```, a vector corresponding to the pixel values from one state of the game. The method returns, ```p```, the probability of taking action 2, to move up. This probability is calculated by running ```logp``` through the sigmoid function. The sigmoid function is a non-linear activation function that returns a value between -1 and 1. A figure of the function is shown below.

<p align = center>
  <img src = figures/sigmoid_function.png>
  </p>

*Fig.4: The sigmoid function between -5 and 5.*

Another thing occurring in this code sample is the calculating of the discounted reward. The reason for this is explained [earlier](#discounted-reward). After taking the dot product from ```W1```, a rectified linear unit (ReLU) function is used to protect weights from fading out because of the lack of a gradient [(Glorot, Bordes, and Bengio, 2011)](#sources). The next thing needed is to backpropagate the policy network to calculate the error gradient. The error value will allow the network to self-correct network weights until desired values (that correspond to maximized discounted reward) are found.

```python
#backpropagation: recursively compute error derivatives for both network layers (W1 and W2)
#programatically the chain rule
#epdlogp: modulate the _ with advantage
def policy_backward(eph,epdlogp):
    #eph is array of intermediate states
    #derivative wrt W2
    dw2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 #reLU
    #derivative wrt W1
    dw1 = np.dot(dh.T, epx)
    #return both derivatives to update weights
    return {'W1':dw1, 'W2':dw2}
```
In ```policy_backward()```, the derivative with respect to ```W1``` is the gradient. The gradient of the weights is the error value, which shows how much the input weights need to be changed to provide the desired output [(Karpathy, 2016)]. Through trial and error (i.e. calling ```policy_forward()``` and ```policy_backward()``` again and again), the calculated gradient incrementally changes the weights of the network until an optimal policy is found (the agent always knows when to move up). 

The last thing left to do is to continually propagate forward and backward through the network, updating the gradient by calling ```policy_backward()```.
```python
   if done:
    episode_number += 1

   
    epx = np.vstack(xs) 
    eph = np.vstack(hs) 
    epdlogp = np.vstack(dlogps) 
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] 

    
    discounted_epr = discount_rewards(epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)


    epdlogp *= discounted_epr
    grad = policy_backward(eph, epdlogp)
```

### Note
For further information on policy gradients, one can refer to David Silver's RL course materials, available [here](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf).


## Sources
[1] Karpathy, A. (2016, May 31). Deep Reinforcement Learning: Pong from Pixels. Retrieved from http://karpathy.github.io/2016/05/31/rl/

[2] Sutton, R. S., & Barto, A. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.

[3] https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

[4] Alzantot, M. (2017, July 09). Deep Reinforcement Learning Demysitifed (Episode 2) - Policy Iteration, Value Iteration and... Retrieved from https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa

[5] Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J.,
and Zaremba, W. OpenAI Gym. arXiv preprint arXiv:1606.01540 (2016).

[6] Silver, D. (n.d.). Lecture 2: Markov Decision Processes. Retrieved from http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/MDP.pdf

[7] Clark, J., & Amodei, D. (2017, March 20). Faulty Reward Functions in the Wild. Retrieved from https://blog.openai.com/faulty-reward-functions/

[8] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., . . . Hassabis, D. (2015, February 25). Human-level control through deep reinforcement learning. Retrieved from https://www.nature.com/articles/nature14236

[9] Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep Sparse Rectifier Neural Networks. Proceedings of the 14th International Conference on Artificial Intelligence and Statistics, 15, 315-323.
