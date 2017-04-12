""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
#matrix math
import numpy as np 
#for loading and saving models. Pickling is the process whereby a Python object hierarchy is converted into a byte stream)
import cPickle as pickle 
#OpenAI's library that provides environments to test RL algorithms in, Universe adds
#even more environments
import gym

#Step 1 - How does pong work?
#1 - receive image frame
#2 - move paddle UP/DOWN? (binary)
#3 - Make action recieve reward (+1 if moves past AI, -1 if missed ball, 0 otherwise)
#lets forget about other pong details. focus on the algo.

#Step 2 - On RL
#1 RL is a Branch of machine learning concerned with taking #sequences of actions
#Usually described in terms of agent interacting with a
#previously unknown environment, trying to maximize cumulative reward
#2 RL combined with other techniques is powerful (AlphaGo, DQN)
#Policy Gradients > DQN (by most ppl, incl. DQN Authors) https://www.youtube.com/watch?v=M8RfOCYIL8k
#3 we'll build a 2 layer fully connected neural network
#https://karpathy.github.io/assets/rl/policy.png
#recieves image pixels, outputs probability of moving UP (stochasticity!)

#Step 3  Policy gradients have 3 key differences from supervised learning
#1 We dont have the correct labels so as a fake label we substitute the action we happened to sample from the policy 
#2 We modulate the loss for each example multiplicatively based on the eventual outcome
#since we want to increase the log probability for actions that worked and decrease it for those that didnt. 
#3 runs on a continuously changing dataset (the episodes), scaled by the advantage, and we only want to do one (or very few) 
#updates based on each sampled dataset.
#4 can be used to backprop when nodes are stochastic (!!, more on this later)

#Step 4 - PG vs Humans
#Policy Gradients have to actually experience a positive reward, and experience it very often in order to eventually and slowly shift the policy parameters towards 
#repeating moves that give high rewards. With our abstract model, humans can figure out what is likely to give rewards without ever actually experiencing the 
#rewarding or unrewarding transition. I dont have to actually experience crashing my car into a wall a few hundred times before I slowly start avoiding to do so.

#Step 5 - Credit Assignment problem
#Credit assignment problem - we just got +1 reward. what caused it? millions parameters and pixels...which frame? 
#no labels!
#https://karpathy.github.io/assets/rl/rl.png
#forward pass > output probability distribution > sample from it to take an action > wait for reward > use reward as gradient
#to update network via backprop
#reward could be any positive number. doesnt matter. its about magnitude. thanks neural nets.

#Training time - 3 nights on a macbook, couple hours on GPU cluster in AWS


# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4 #for convergence (too low- slow to converge, too high,never converge)
gamma = 0.99 # discount factor for reward (i.e later rewards are exponentially less important)
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid (the pong world)
if resume:
  model = pickle.load(open('save.p', 'rb')) #load from pickled checkpoint
else:
  model = {} #initialize model 
  #rand returns a sample (or samples) from the standard normal distribution
  #xavier algo determines the scale of initialization based on the number of input and output neurons.
  #Imagine that your weights are initially very close to 0. What happens is that the signals shrink as it goes through each 
  #layer until it becomes too tiny to be useful. Now if your weights are too big, the signals grow at each layer 
  #it passes through until it is too massive to be useful.
  #By using Xavier initialization, we make sure that the weights are not too small but not too big to propagate accurately the signals.
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
  #zeros like returns an array of zeros with the same shape and type as a given array.
  #we will update buffers that add up gradients over a batch
  #where the model contains kv pairs, weights layers etc 
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } 
## rmsprop (gradient descent) memory used to update model
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } 

#activation function
def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

#takes a single game frame as input
#preprocesses before feeding into model
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel() #flattens 

#In practice the discount factor could be used to model the fact that the decision maker is uncertain 
#about if in the next decision instant the world (e.g., environment / game / process ) is going to end.
#If the decision maker is a robot, the discount factor could be the probability that the robot is switched 
#off in the next time instant (the world ends in the previous terminology). That is the reason why the robot 
#is short sighted and does not optimize the sum reward but the discounted sum reward.

#since we are otimizing for short term reward, not long term reward (like legend of zelda)
def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  #initilize discount reward matrix as empty
  discounted_r = np.zeros_like(r)
  #to store reward sums
  running_add = 0
  #for each reward
  for t in reversed(xrange(0, r.size)):
    #if reward at index t is nonzero, reset the sum, since this was a game boundary (pong specific!)
    if r[t] != 0: running_add = 0 
    #increment the sum 
    #https://github.com/hunkim/ReinforcementZeroToAll/issues/1
    running_add = running_add * gamma + r[t]
    #earlier rewards given more value over time 
    #assign the calculated sum to our discounted reward matrix
    discounted_r[t] = running_add
  return discounted_r


#forward propagation via numpy woot!
def policy_forward(x):
  #matrix multiply input by the first set of weights to get hidden state
  #will be able to detect various game scenarios (e.g. the ball is in the top, and our paddle is in the middle)
  h = np.dot(model['W1'], x)
  #apply an activation function to it
  #f(x)=max(0,x) take max value, if less than 0, use 0
  h[h<0] = 0 # ReLU nonlinearity
  #repeat process once more
  #will decide if in each case we should be going UP or DOWN.
  logp = np.dot(model['W2'], h)
  #squash it with an activation (this time sigmoid to output probabilities)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  #recursively compute error derivatives for both layers, this is the chain rule
  #epdlopgp modulates the gradient with advantage
  #compute updated derivative with respect to weight 2. It's the parameter hidden states transpose * gradient w/ advantage (then flatten with ravel())
  dW2 = np.dot(eph.T, epdlogp).ravel()
  #Compute derivative hidden. It's the outer product of gradient w/ advatange and weight matrix 2 of 2
  dh = np.outer(epdlogp, model['W2'])
  #apply activation
  dh[eph <= 0] = 0 # backpro prelu
  #compute derivative with respect to weight 1 using hidden states transpose and input observation
  dW1 = np.dot(dh.T, epx)
  #return both derivatives to update weights
  return {'W1':dW1, 'W2':dW2}

#environment
env = gym.make("Pong-v0")
#Each timestep, the agent chooses an action, and the environment returns an observation and a reward.
#The process gets started by calling reset, which returns an initial observation
observation = env.reset()
prev_x = None # used in computing the difference frame
#observation, hidden state, gradient, reward
xs,hs,dlogps,drs = [],[],[],[]
#current reward
running_reward = None
#sum rewards
reward_sum = 0
#where are we?
episode_number = 0

#begin training!
while True:

  # preprocess the observation, set input to network to be difference image
  #Since we want our policy network to detect motion
  #difference image = subtraction of current and last frame
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x
  #so x is our image difference, feed it in!

  # forward the policy network and sample an action from the returned probability
  aprob, h = policy_forward(x)
  #this is the stochastic part 
  #since not apart of the model, model is easily differentiable
  #if it was apart of the model, we'd have to use a reparametrization trick (a la variational autoencoders. so badass)
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  env.render()
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    #each episode is a few dozen games
    epx = np.vstack(xs) #obsveration
    eph = np.vstack(hs) #hidden
    epdlogp = np.vstack(dlogps) #gradient
    epr = np.vstack(drs) #reward
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    #the strength with which we encourage a sampled action is the weighted sum of all rewards afterwards, but later rewards are exponentially less important
    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    #advatnage - quantity which describes how good the action is compared to the average of all the action.
    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    #http://68.media.tumblr.com/2d50e380d8e943afdfd66554d70a84a1/tumblr_inline_o4gfjnL2xK1toi3ym_500.png
    if episode_number % batch_size == 0:
      for k,v in model.iteritems():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
    print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')
