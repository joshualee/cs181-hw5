

# Components of a darts player. #

# 
 # Modify the following functions to produce a player.
 # The default player aims for the maximum score, unless the
 # current score is less than or equal to the number of wedges, in which
 # case it aims for the exact score it needs.  You can use this
 # player as a baseline for comparison.
 #

from random import *
import throw
import darts

# make pi global so computation need only occur once
PI = {}
EPSILON = .001


# actual
def start_game(gamma):

  infiniteValueIteration(gamma)
  #for ele in PI:
    #print "score: ", ele, "; ring: ", PI[ele].ring, "; wedge: ", PI[ele].wedge
  
  return PI[throw.START_SCORE]

def get_target(score):

  return PI[score]

def get_wedge_location(wedge):
  return throw.wedges.index(wedge)

def get_adj_wedge(wedge, i):
  w_loc = get_wedge_location(wedge)
  return throw.wedges[(w_loc + i) % throw.NUM_WEDGES]

# define transition matrix/ function
def T(a, s, s_prime):  
  # takes an action a, current state s, and next state s_prime
  # returns the probability of transitioning to s_prime when taking action a in state s
  target = s - s_prime
  target_locations = []
  p = 0.0
  
  # find all wedge/ring combos that would lead to s -> s' transition
  for i in range(-2, 3):
    current_wedge = get_adj_wedge(a.wedge, i)
    
    # iterate through all possible rings
    for j in range(-2, 3):
      ring = a.ring + j
      
      # off dart board
      if ring >= throw.MISS:
        continue
      
      # allow for ring "wrap around", e.g. the ring inside and outside the center
      # ring is the inner ring
      if ring < 0:
        ring = abs(ring)
        
      new_location = throw.location(ring, current_wedge)
      
      # hitting target would go from s -> s'!
      if target == throw.location_to_score(new_location):
        # calculate probability of hitting target
        if i == 0:
          w_p = 0.4
        elif abs(i) == 1 :
          w_p = 0.2
        elif abs(i) == 2:
          w_p = 0.1
        else:
          assert False, "Impossible wedge"

        if j == 0:
          r_p = 0.4
        elif abs(j) == 1 :
          r_p = 0.2
        elif abs(j) == 2:
          r_p = 0.1
        else:
          assert False, "Impossible ring"
        
        p += (w_p * r_p)
  
  return p

def infiniteValueIteration(gamma):
  # takes a discount factor gamma and convergence cutoff epislon
  # returns

  V = {}
  Q = {}
  V_prime = {}
  
  states = darts.get_states()
  actions = darts.get_actions()

  notConverged = True

  # intialize value of each state to 0
  for s in states:
    V[s] = 0
    Q[s] = {}

  # until convergence is reached
  while notConverged:

    # store values from previous iteration
    for s in states:
      V_prime[s] = V[s]

    # update Q, pi, and V
    for s in states:
      for a in actions:

        # given current state and action, sum product of T and V over all states
        summand = 0
        for s_prime in states:
          summand += T(a, s, s_prime)*V_prime[s_prime]

        # update Q
        Q[s][a] = darts.R(s, a) + gamma*summand

      # given current state, store the action that maximizes V in pi and the corresponding value in V
      PI[s] = actions[0]
      
      # bug fix from piazza post 283
      V[s] = Q[s][PI[s]] 
      
      for a in actions:
        if V[s] <= Q[s][a]:
          V[s] = Q[s][a]
          PI[s] = a

    notConverged = False
    for s in states:
      if abs(V[s] - V_prime[s]) > EPSILON:
        notConverged = True