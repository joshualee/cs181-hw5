import random
import throw
import darts
import math

EPSILON_VI = .001

# NOTE: We did not implement the interface methods start_game and get_target.
# Functions below are simply the default darts player discussed in modelfree.py.
# Instead, we defined the function modelbased, which uses the helper methods
# ex_strategy_one and ex_strategy_two.

def start_game():

  return(throw.location(throw.INNER_RING, throw.NUM_WEDGES)) 

def get_target(score):

  if score <= throw.NUM_WEDGES: return throw.location(throw.SECOND_PATCH, score)
  
  return(throw.location(throw.INNER_RING, throw.NUM_WEDGES))


def exp_cooling_function(t):
  return 50.0 * pow(math.e, -t / 250.0)

# Define your first exploration/exploitation strategy here. Return 0 to exploit and 1 to explore. 
# You may want to pass arguments from the modelbased function. 
def ex_strategy_one(actions, pi_star, s, trials):
  epsilon = 1.0 / trials
  u = random.uniform(0, 1)
  if u <= epsilon:
    # explore
    return random.randint(0, len(actions)-1)
  else:
    # exploit
    return pi_star[s]

# Define your first exploration/exploitation strategy here. Return 0 to exploit and 1 to explore. 
# You may want to pass arguments from the modelbased function.
def ex_strategy_two(actions, Q, s, trials):
  # uniformly choose random action if we have not estimated Q yet
  if Q == {}:
    return random.randint(0, len(actions)-1)
  
  # calculate probabilities according to Boltzmann exploration
  temp = exp_cooling_function(trials)
  p = {}
  total_p = 0.0
  for a in range(len(actions)):
    p[a] = pow(math.e, Q[s][a] / temp)
    total_p += p[a]
  
  # normalize the probability
  for a in range(len(actions)):
    p[a] = p[a] / total_p
  
  # random draw
  u = random.uniform(0, 1)
  
  # pick action a with probability p[a]
  current_p = 0.0
  for a in range(len(actions)):
    current_p += p[a]
    if u < current_p:
      return a
      
# Implement a model-based reinforcement learning algorithm. 
# Given num_games (the number of games to play), store the
# learned transition probabilities in T.
def modelbased(gamma, epoch_size, num_games):

    # store all actions (targets on dartboard) in actions array
    actions = darts.get_actions()
    states = darts.get_states()
    
    pi_star = {}
    g = 0
    num_actions = {}
    num_transitions = {}
    T_matrix = {}
    num_iterations = 0
    
    # initialize v
    V = {}
    V[0] = {}
    V[1] = {}
    for s in states:
      V[0][s] = 0
      V[1][s] = 0
    
    # Initialize all arrays to 0 except the policy, which should be assigned a random action for each state.
    for s in states:
        pi_star[s] = random.randint(0, len(actions)-1)
        num_actions[s] = {}
        num_transitions[s] = {}
        T_matrix[s] = {}
        
        for a in range(len(actions)):
            num_actions[s][a] = 0

        for s_prime in states:
            num_transitions[s][s_prime] = {}
            T_matrix[s][s_prime] = {}
            for a in range(len(actions)):
                num_transitions[s][s_prime][a] = 0
                T_matrix[s][s_prime][a] = 0


    # play num_games games, updating policy after every EPOCH_SIZE number of throws
    for g in range(1, num_games + 1):
        iterations_this_game = 0
        Q = {}
    
    	# run a single game
        s = throw.START_SCORE
        while s > 0:
            iterations_this_game += 1
            num_iterations += 1
    		
            # The following two statements implement two exploration-exploitation
            # strategies. Comment out the strategy that you wish not to use.
            
            # a = ex_strategy_one(actions, pi_star, s, iterations_this_game)
            a = ex_strategy_two(actions, Q, s, iterations_this_game)
            action = actions[a]
            
            # Get result of throw from dart thrower; update score if necessary
            loc = throw.throw(action) 
            s_prime = s - throw.location_to_score(loc)
            if s_prime < 0:
                s_prime = s
                
            # Update experience:
            # increment number of times this action was taken in this state;
            # increment number of times we moved from this state to next state on this action.

            num_actions[s][a] += 1
            num_transitions[s][s_prime][a] += 1

	    # Next state becomes current state 
            s = s_prime

            # Update our learned MDP and optimal policy after every EPOCH_SIZE throws, 
            # using infinite-horizon value iteration. 
		
            if num_iterations % epoch_size == 0:

                # Update transition probabilities
                for i in states:
                    for j in states:
                        for k in range(len(actions)):
                            if num_actions[i][k] != 0:
                                T_matrix[i][j][k] = float(num_transitions[i][j][k]) / float(num_actions[i][k])

                # Update strategy (stored in pi) based on newly updated reward function and transition
                # probabilities 
                T_matrix, pi_star, Q, V = modelbased_value_iteration(gamma, T_matrix, pi_star, actions, states, V)
    
    avg_turns = float(num_iterations)/float(num_games)
    print "Average turns = ", avg_turns
    return avg_turns

# A modified version of infinite horizon value iteration from part 2 */
def modelbased_value_iteration(gamma, T_matrix, pi_star, actions, states, V):
  converging = 0
  num_iterations = 0
  Q = {}
  
  # iterate until all state values (v[s]) converge 
  while not(converging):
    num_iterations += 1
    for s in states:
      Q[s] = {}
      for a in range(len(actions)):

        # find the value of each action, given state s 
        Q[s][a] = darts.R(s, actions[a])
        for s_prime in states:

          Q[s][a] += gamma * T_matrix[s][s_prime][a] * V[0][s_prime]

          # find the action that maximizes Q and the maximum value of Q
          if a == 0 or (Q[s][a] >= V[1][s]):
            pi_star[s] = a
            V[1][s] = Q[s][a]

                  
    # values of v for iteration k become the values of v for iteration k-1
    converging = True
    for s in states:
      # check for one component that does not converge
      if EPSILON_VI < abs(V[0][s] - V[1][s]):
        converging = False

      V[0][s] = V[1][s]
  
  return T_matrix, pi_star, Q, V