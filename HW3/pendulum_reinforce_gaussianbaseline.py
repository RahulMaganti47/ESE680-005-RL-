import numpy as np
import scipy.signal
import gym
import pdb
import matplotlib.pyplot as plt
from pendulum import *
import pdb 

class ContinuousPolicy(object):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states 
        self.num_actions = num_actions
        # here are the weights for the policy - you may change this initialization       
        self.weights = np.zeros((self.num_states, self.num_actions))


    # TODO: fill this function in    
    # it should take in an environment state
    def act(self, state):
        pass

    # TODO: fill this function in    
    # computes the gradient of the discounted return    
    # at a specific state and action    
    # use the computed advantage function appropriately.
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, advantage):
        pass

    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())    
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        pass

#design linear baseline
class LinearValueEstimator(object):
    def __init__(self, num_states):
        self.num_states = num_states
        
	self.weights = np.zeros(num_states)


    # TODO: fill this function in
    #takes in a state and predicts a value for the state
    def predict(self,state):
        pass

    # TODO: fill this function in
    # construct a suitable loss function and use it to update the 
    # values of the value estimator. choose suitable step size for updating the value estimator
    def update(self, state, value_estimate, target, value_step_size):
        pass


# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]
def get_discounted_returns(rewards, gamma):
    pass

# TODO: fill this function in 
# this will take in an environment, GridWorld
# a continuous policy 
# a value estimator,
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
# make sure to add in the baseline computation here. 
# Using the computed baseline, compute the advantage. 
# Use this advantage in the policy gradient calculation
def reinforce(env, policy,value_estimator):
        pass

if __name__ == "__main__":
    env = Continuous_Pendulum()

    # TODO: define num_states and num_actions
    policy = ContinuousPolicy(num_states, num_actions)
    value_estimator = LinearValueEstimator(num_states)
    reinforce(env, policy, value_estimator)

    # Test time
    state = env.reset()
    env.print()
    done = False
    state_hist = []
    while not done:
        input("press enter:")
        action = policy.act(state)
        state, reward, done = env.step(action)
        state_hist.append(state)

    # Plotting test time results
    state_hist = np.array(state_hist)
    plt.plot(state_hist[0, :])
    plt.xlabel("time (s)")
    plt.ylabel("angle (rad)")
    plt.show()
