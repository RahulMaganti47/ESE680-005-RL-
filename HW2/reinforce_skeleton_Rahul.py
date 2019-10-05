from grid_world import *
import numpy as np
from scipy.special import softmax
import copy 
import matplotlib.pyplot as plt 
import pdb 

class DiscreteSoftmaxPolicy(object):
    def __init__(self, num_states, num_actions, temperature):
        self.num_states = num_states
        self.num_actions = num_actions
        self.temperature = temperature

        # here are the weights for the policy
        #self.weights = np.arange(num_states*num_actions).reshape((num_states, num_actions))
        self.weights = np.zeros((num_states, num_actions))
        #inter = np.arange(start=0, stop=16, step=1, dtype='float')  
        #self.weights = inter.reshape(num_states, num_actions)

    def r_softmax(self, x): 
        return np.exp(x) / np.sum(np.exp(x)) 
   
    def get_logits(self, state):
        logits = self.r_softmax(self.weights[state, :])
        return logits

    # TODO: fill this function in
    # it should take in an environment state
    # return the action that follows the policy's distribution
    def act(self, state):   
        actions_a =  self.get_logits(state) 
        return np.random.choice(self.num_actions, 1, p=actions_a)[0] 

    # TODO: fill this function in
    # computes the gradient of the discounted return 
    # at a specific state and actio
    # return the gradient, a (self.num_states, self.num_actions) numpy array 
    
    def compute_gradient(self, state, action, discounted_return): 
        grad = np.zeros((self.num_states, self.num_actions)) 
        policy_a = self.get_logits(state)
        one_hot = np.zeros(policy_a.shape)
        one_hot[action] = 1.0
        d_sa = (one_hot - policy_a)
        grad[state, :] = d_sa * discounted_return
                          
        return grad

    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size): 
        self.weights += step_size*grad 

# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]

def discounts_rewards(rewards, gamma, time_steps):
    moving_addition = 0
    discounted_returns = np.zeros(time_steps)
    for i in reversed(range(0, len(rewards))):
        moving_addition = moving_addition * gamma + rewards[i] 
        discounted_returns[i] = moving_addition
    return discounted_returns

# TODO: fill this function in 
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
def reinforce(env, policy, gamma, num_episodes, learning_rate): 
    rewards_sum = []
    episodes = []
    for i in range(num_episodes): 
        rewards = []
        states = []  
        actions = []
        done = False
        state = env.reset() 
        time_steps = 0        
        score = 0 
    
        while not done: 
            policy_a = policy.act(state)
            next_state, reward, done = env.step(policy_a) 
            actions.append(policy_a)
            rewards.append(reward)
            states.append(state)
            state = copy.deepcopy(next_state) 
            time_steps = time_steps + 1  
            score += reward 
        
        if ((i % 10) == 0): 
            print(score)
        if ((i % 100) == 0): 
            rewards_sum.append(score)
            episodes.append(i)

        gradient = np.zeros(policy.weights.shape) 
        discounted_r_array = discounts_rewards(rewards, gamma, time_steps)
        
        for t in range(time_steps):  
            gradient += policy.compute_gradient(states[t], actions[t], discounted_r_array[t])
        
        policy.gradient_step(gradient, learning_rate)

    plt.plot(episodes, rewards_sum)
    
    return policy.weights

if __name__ == "__main__":
    gamma = 0.9 
    num_episodes = 20000
    learning_rate = 1e-4
    env = GridWorld(MAP2) 
    policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions(), temperature=1) 
    reinforce(env, policy, gamma, num_episodes, learning_rate)
    plt.xlabel("Episode #")
    plt.ylabel("Sum of rewards")
    plt.title("Episodes vs. Sum of Rewards [MAP2]")
    plt.show()
    
    # gives a sample of what the final policy looks like 
    print("Rolling out final policy")
    state = env.reset()
    env.print_()
    done = False
    while not done:
        #input("press enter to continue:")
        action = policy.act(state)
        state, reward, done = env.step(action)
        env.print_()

