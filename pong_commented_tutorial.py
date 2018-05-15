# Code from Karpathy's "Pong From Pixels" with additional
# comments to explain what different parts of the program do.


import numpy as np #matrix math
import pickle #serializing data (save/load model)
import gym #atari environment


# no hard coded rules
# algorithm, not environment
# 1) receive image from the game (game STATE)
# 2) binary decision - move paddle up or down
# 3) make an action, receive a reward
# Action list: +1 for getting ball past AI,
# -1 for letting ball go past, 0 for any other action

# general algorithm for any game
# 2 layer neural network that takes in frames of the game (STATE),
# output is a probablility value of whether to move up or down
# we sample from probability value to get POLICY, getting gradient
# as we backpropagate.

#stochastic: non-deterministic, unpredictable, random, making decisions that are NOT predetermined.
#adding variation into networks to try and mimic human behavior.
#gradients == partial derivatives


#hyperparameters
H = 200 #number of hidden neurons
batch_size = 10 #number of episodes in a parameter update
learning_rate = 0.0001 #learning_rate
gamma = 0.99 #discount factor (later rewards are less important, optimizing for short term)
decay_rate = 0.99 #RMSprop
resume = False

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

#preprocessing function: converts game image frame I into paddles and ball
def prepro(I):
    I = I[35:195] #cropping the game frame
    I = I[::2, ::2, 0] #downsampling by factor of 2
    I[I == 144] = 0 #erase background layer 1
    I[I == 109] = 0 #erase background layer 2
    I[I != 0] = 1 #paddles and balls set to 1
    return I.astype(np.float).ravel() #flatten

#optimizing for short term rewards by weighing each reward differently by how early they occurred
#source: https://github.com/hunkim/ReinforcementZeroToAll/issues/1
#weighing immediate rewards higher than later rewards exponentially
#short term: did the ball go past the AI paddle?
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
    return p, h #return propability of taking action 2 and hidstate

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

#implementation details
env = gym.make('Pong-v0')
observation = env.reset()
prev_x = None
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
print("Number of states")
print(env.observation_space.n)


while True:


  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x



  aprob, h = policy_forward(x)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!


  xs.append(x)
  hs.append(h)
  y = 1 if action == 2 else 0
  dlogps.append(y - aprob)

  #env.render()
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward)

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
    for k in model: grad_buffer[k] += grad[k] 

    
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] 
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v)

    
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset()
    prev_x = None

print(env.observation_space.n)
if reward != 0:
    print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !')
