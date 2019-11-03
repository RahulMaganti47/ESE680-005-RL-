from grid_world import *
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


class DiscreteSoftmaxPolicy(object):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states 
        self.num_actions = num_actions
        # here are the weights for the policy - you may change this initialization      
        self.weights = np.zeros((self.num_states, self.num_actions))

    # TODO: fill this function in    
    # it should take in an environment state

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    
    def get_logits(self, state):
        actions = self.softmax(self.weights[state, :])
        return actions
        
    def act(self, state):
        actions_a = self.get_logits(state)
        action = np.random.choice(self.num_actions, 1, p=actions_a)[0]
        return action 

    # TODO: fill this function in    
    # computes the gradient of the discounted return    
    # at a specific state and action    
    # use the computed advantage function appropriately.
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action,advantage):
        grad = np.zeros(self.num_states, self.num_actions)
        possible_actions = self.get_logits(state)
        one_hot = np.zeros(possible_actions.shape)
        one_hot[action] = 1.0
        d_sa = one_hot - possible_actions
        grad[state, :] = d_sa * advantage 
        return grad 
    
        
    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())    
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.weights += grad*step_size


class ValueEstimator(object):
    def __init__(self, num_states):
        self.num_states = num_states
        
        #initial value estimates or weights of the value estimator are set to zero. 
        self.values = np.zeros((self.num_states))
    # TODO: fill this function in
    #takes in a state and predicts a value for the state
    def predict(self,state): 
        return self.values[state]

    # TODO: fill this function in
    # construct a suitable loss function and use it to update the 
    # values of the value estimator. choose suitable step size for updating the value estimator
    def update(self, state, value_estimate, target, value_step_size):
        difference = target - value_estimate
        squared_difference = np.square(difference)
        N = len(squared_difference)
        loss = np.sum(squared_difference)
        mean_loss = (1/N) * loss
        self.values[state] += value_step_size * mean_loss

# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]

def get_discounted_returns(rewards, gamma, length_traj):
    moving_add = 0
    discounted_rewards = np.zeros(length_traj)
    for i in reversed(range(0, len(rewards))): 
        moving_add = moving_add * gamma + rewards[i]
        discounted_rewards[i] =moving_add

    return discounted_rewards
# TODO: fill this function in 
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a value estimator,
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
# make sure to add in the baseline computation here. 
# Using the computed baseline, compute the advantage. 
# Use this advantage in the policy gradient calculation
def reinforce(env, policy, value_estimator, num_episodes, gamma, alpha_p, alpha_b):

    for ep in range(num_episodes): 
        rewards = []
        states = []
        actions = [] 
        state = env.reset()
        done = False
        length_traj = 0 
        score = 0 
        print("Episode #: {}, Score: {} ", ep, score)
    
        while not done:
            policy_a = policy.act(state)
            next_state, reward, done = env.step(policy_a)
            rewards.append(reward)
            states.append(state)
            actions.append(actions)
            state = next_state
            length_traj += 1
            score += reward 
    
        discounted_rewards = policy.discounted_rewards(rewards, gamma, length_traj)
        
        for t in range(length_traj):
            advantage = discounted_rewards[t] - value_estimator.predict(state[t])
            grad_p = policy.compute_gradient(state[t], action[t], advantage)
            policy.gradient_step(grad_p, alpha_p)       
            value_estimator.update(state[t], value_estimator.predict(state[t]), discounted_rewards[t], alpha_b)
        
        
    

if __name__ == "__main__":
    env = GridWorld(MAP1)
    env.print()
    num_episodes = 1000
    gamma = .97 
    alpha_p = .01
    alpha_b = .001
    #temperature must be tuned.
    policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions())
    value_estimator = ValueEstimator(env.get_num_states())
    reinforce(env, policy,value_estimator, num_episodes, gamma,alpha_p, alpha_b)

    #Test time
    state = env.reset()
    env.print()
    done = False
    while not done:
        input("press enter:")
        action = policy.act(state)
        state, reward, done = env.step(action)
        env.print()


