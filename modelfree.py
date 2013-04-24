from random import *
import throw
import darts
import math
 
# The default player aims for the maximum score, unless the
# current score is less than the number of wedges, in which
# case it aims for the exact score it needs. 
#  
# You may use the following functions as a basis for 
# implementing the Q learning algorithm or define your own 
# functions.

GAMMA = 0.5 # discount factor
ALPHA = 0.5 # learning rate

Q = {}
states = None
actions = None

cur_s = throw.START_SCORE # current state
last_a = None
throws = 0

def start_game():
  global Q, states, actions, cur_s, last_a, throws
  
  states = darts.get_states()
  actions = darts.get_actions()
  cur_s = throw.START_SCORE
  throws = 1
  
  Q = {}
  for s in states:
    Q[s] = {}
    for a in actions:
      Q[s][a] = 0.0
  
  # start by returning uniform random action
  last_a = choice(actions)
  return last_a

def get_target(new_s):
  global Q, states, actions, cur_s, last_a, throws
  
  # increment number of throws
  throws += 1
  
  # calculate how much reward we got going from state s to state score
  # (this is just how many points we got)
  reward = cur_s - new_s
  
  # find max over actions a' : Q(s', a') where s' is the new state
  max_Qsa = get_max_Qsa(Q, new_s)
  
  # update Q
  Q[cur_s][last_a] += ALPHA * (reward + GAMMA * max_Qsa - Q[cur_s][last_a])

  # update current state
  cur_s = new_s
  
  # last_a = ex_strategy_one(actions, throws, Q, cur_s)
  last_a = ex_strategy_two(actions, throws, Q, cur_s)
  return last_a

def get_argmax_Qsa(Q, s):
  best_a = None
  max_Qsa = float("-inf")
  for a in actions:
    new_max = Q[s][a]
    if new_max > max_Qsa:
      max_Qsa = new_max
      best_a = a
  return best_a

def get_max_Qsa(Q, s):
  return Q[s][get_argmax_Qsa(Q, s)]
  
# Exploration/exploitation strategy one.
def ex_strategy_one(actions, throws, Q, cur_s):
  epsilon = 1.0 / throws
  u = uniform(0, 1)
  if u <= epsilon:
    # explore
    return choice(actions)
  else:
    # exploit
    return get_argmax_Qsa(Q, cur_s)

def exp_cooling_function(t):
  return 50.0 * pow(math.e, -t / 250.0)

# Exploration/exploitation strategy two.
def ex_strategy_two(actions, throws, Q, cur_s):
  # calculate probabilities according to Boltzmann exploration
  temp = exp_cooling_function(throws)
  p = {}
  total_p = 0.0
  for a in actions:
    p[a] = pow(math.e, Q[cur_s][a] / temp)
    total_p += p[a]

  # normalize the probability
  for a in actions:
    p[a] = p[a] / total_p

  # random draw
  u = uniform(0, 1)

  # pick action a with probability p[a]
  cur_p = 0.0
  for a in actions:
    cur_p += p[a]
    if u < cur_p:
      return a