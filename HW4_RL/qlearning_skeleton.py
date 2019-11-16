import numpy as np
from grid_world import *
import matplotlib.pyplot as plt
import random 
# TODO: Fill this function in
# Function that takes an a 2d numpy array Q (num_states by num_actions)
# an epsilon in the range [0, 1] and the state
# to output actions according to an Epsilon-Greedy policy
# (random actions are chosen with epsilon probability)

class QLearning(object):
    # Initialize a Qlearning object
    # alpha is the "learning_rate"
    def __init__(self, num_states, num_actions, eps, learning_rate=0.9, discount_factor=.999):
         # initialize Q values to something
        self.num_states = num_states
        self.num_actions = num_actions  
        
        #initilaize the q_table                                                                                                                                                         
        self.q_table = np.random.random_sample([self.num_states, self.num_actions])
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eps = eps 
    
    def tabular_epsilon_greedy_policy(self, eps, state):

        actions = self.q_table[state]
        max_action = np.argmax(actions) 
        num_actions = len(actions)
        rand_action = np.random.choice(num_actions)
        temp_rand = random.uniform(0, 1)
        if (eps > temp_rand):
            return max_action
        else: 
            return rand_action

    # TODO: fill in this function
    # updates the Q value table
    # with a (state, action, reward, next_state) from one step in the environment
    # done is a bool indicating if the episode terminated at this step
    # you can return anything from this function to help with plotting/debugging
    def update(self, state, action, reward, next_state, done):
        #get the q_values for the next state
        q_next = self.q_table[next_state]
        #set the q_next values to 0 if we terminate the episode 
        q_next = np.zeros([self.num_actions]) if done else q_next 
        
         #update equation: reward + gamma*max(q_next)
        q_target = reward + self.discount_factor*np.max(q_next)

        # perform the bootstrapped udpate
        self.q_table[state, action] += self.learning_rate*(q_target - self.q_table[state, action])

        # apply exponential decay to epsilon
        if done:
            self.eps = self.eps * .99

                
        return self.q_table 

# TODO: fill this in
# run the greedy policy (no randomness) on the environment for niter number of times
# and return the fraction of times that it reached the goal
def evaluate_greedy_policy(qlearning, env, eps, niter=100): 
    
    num_times_reached = 0
    for iter in range(niter):
        state = env.reset()
        done = False
        score = 0 
        while not done: 
            actions = qlearning.q_table[state]
            (next_state, reward, done) = env.step(action) 
            state = next_state
            score += reward 
        
        if (score > 0): num_times_reached +=1 

    frac = num_times_reached / niter
    return frac 

if __name__ == "__main__":
    env = GridWorld(MAP3)
    eps = 1.0 

    qlearning = QLearning(env.get_num_states(), env.get_num_actions(), eps)
 
    ## TODO: write training code here
    num_episodes = 1000 
    num_states = env.get_num_states() 
    total_reward =   0 
    
    for ep in range(num_episodes):    
        state = env.reset() 
        score = 0 
        done = False 
        while not done:
            action = qlearning.tabular_epsilon_greedy_policy(eps, state)    
            (next_state, reward, done) = env.step(action)
            q_table = qlearning.update(state, action, reward, next_state, done)
            #print(q_table)
            state = next_state
            print("state {} action {}".format(state, action))
            score += reward
        
        print("Episode: {}, Score: {}".format(ep, score))
        total_reward += score 
        
    print("Total Reward: {}".format(total_reward)) 
    
    # evaluate the greedy policy to see how well it performs
    frac = evaluate_greedy_policy(qlearning, env, eps)
    print("Finding goal " + str(frac) + "% of the time.")
