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

def start_game():

  return(throw.location(throw.INNER_RING, throw.NUM_WEDGES)) 

def get_target(score):

  if score <= throw.NUM_WEDGES: return throw.location(throw.SECOND_PATCH, score)
  
  return(throw.location(throw.INNER_RING, throw.NUM_WEDGES))


# Exploration/exploitation strategy one.
def ex_strategy_one():
  return 0


# Exploration/exploitation strategy two.
def ex_strategy_two():
  return 1


# The Q-learning algorithm:
def Q_learning():
  gamma = 0.5
  alpha = 0.5
  
  states = darts.get_states()
  actions = darts.get_actions()
  
  Q = {}
  
  for s in states:
    Q[s] = {}
    for a in actions:
      Q[s][a] = 0.0
  
  converged = False
  while not converged:
    
    action = ex_strategy_one()
    # action = ex_strategy_two()
    
    max_Qsa = float("-inf")
    for a in actions:
      score = throw.location_to_score(a)
      new_max = Q[s - score][a] # s' = s - score
      if new_max > max_Qsa:
        max_Qsa = new_max
    
    reward = darts.R(s, action)
    Q[s][action] = Q[s][action] + alpha * (reward + gamma * max_Qsa - Q[s][a])
