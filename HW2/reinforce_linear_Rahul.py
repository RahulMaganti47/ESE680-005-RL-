from mountain_car import *
import numpy as np
import pdb
from scipy.stats import truncnorm 

class LinearPolicy(object):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        # here are the weights for the policy - you may change this initialization
        self.K = np.zeros(2)
        self.sigma = .01
        
    # TODO: fill this function in 
    # it should take in an environment state
    # return the action that follows the policy's distribution
    
    def get_truncated_normal(self, mean, sd, low, upp):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def act(self, state): 
        #pdb.set_trace()
        mu = np.dot(self.K, state) 
        probs_a = self.get_truncated_normal(mu, self.sigma, -1, 1)
        action = probs_a.rvs()
        act_a = np.array([action])
        #action = np.random.normal(mu, self.sigma, 1) 
        #if (action[0] > 1):
        #    action[0] = 1
        #elif (action[0] < -1): 
        #    action[0] = -1 
        return act_a
    
    # TODO: fill this function in
    # computes the gradient of the discounted return 
    # at a specific state and action
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, discounted_return):
        grad = np.zeros(len(state)) 
        mean = np.dot(self.K, state) 
        d_sa = np.zeros(len(state), dtype='float64')
        d_sa[0] = (action - mean) * state[0]
        d_sa[1] = (action - mean) * state[1] 
        d_scaled = d_sa /(self.sigma**2) 
        #pdb.set_trace()
        grad = d_scaled*discounted_return
        return grad


    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.K += step_size*grad


# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]
def get_discounted_returns(rewards, gamma, time_steps):
    moving_add = 0 
    discounted_rewards = np.zeros(time_steps)
    for i in reversed(range(0, len(rewards))):
        moving_add = gamma*moving_add + rewards[i]
        discounted_rewards[i] = moving_add
    
    return discounted_rewards

# TODO: fill this function in 
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
def reinforce(env, policy, gamma, num_episodes, learning_rate):
    for i in range(num_episodes):
        rewards = []
        states = []
        actions = [] 
        done = False 
        state = env.reset() 
        t = 0 
        score = 0
        print("Episode # ", i)
        while not done:

            a = policy.act(state)
            next_state, reward, done, _ = env.step(a)
            rewards.append(reward)
            states.append(next_state)
            actions.append(a)
            score += reward 
            #print(score)
            state = next_state
            t += 1
        num_time_steps = t  
        discounted_return = get_discounted_returns(rewards, gamma, num_time_steps) 
        grad = np.zeros(policy.K.shape)
        for t in range(num_time_steps):
            grad += policy.compute_gradient(states[t], actions[t], discounted_return[t])
            print("gradient: ", grad)
        policy.gradient_step(grad, learning_rate)
         
    return policy.K 


if __name__ == "__main__": 
    gamma = 0.9
    num_episodes = 1000
    learning_rate = 1e-4
    env = Continuous_MountainCarEnv()
    
    policy = LinearPolicy(2, 1)
    reinforce(env, policy, gamma, num_episodes, learning_rate)

    # gives a sample of what the final policy looks like
    print("Rolling out final policy")
    state = env.reset()
    env.print_()
    done = False
    while not done:

        #input("press enter to continue:")
        action = policy.act(state)
        state, reward, done, _ = env.step([action])
        env.print_()
