from pendulum import * 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable 
from torch.distributions import Normal
from decimal import Decimal
import copy as cp
 

class ContinuousPolicy(nn.Module):
    def __init__(self, num_states, num_actions, seed): 
        super(ContinuousPolicy, self).__init__()   
        self.num_states = num_states
        self.num_actions = num_actions
        np.random.seed(seed)
    
        #self.K = np.random.randn(3)
        #self.sigma = 1
         
        self.layer1 = nn.Linear(self.num_states, 20, bias=False)
        self.action_mu = nn.Linear(20, self.num_actions, bias=False) 
        self.policy_history = torch.Tensor(()).double()
        

        self.model = torch.nn.Sequential( 
            self.layer1,
            nn.Dropout(p=0.6),
              nn.ReLU(),
            self.action_mu,
            nn.Softmax(dim=-1)  
        ) 

    
    def forward(self, state):
        return self.model(state)
 
# TODO: fill this function isn  
# it should take in an environment state 
def act(state, policy):
    #mu= np.dot(state, self.K)
    #action = np.random.normal(mu, self.sigma)
    
    #return action 
    # pylint: disable=E1101
    state = torch.from_numpy(state).type(torch.double)
    # pylint: enable=E1101 
    action_mean = policy(Variable(state))
    #dist = Normal(action_mean, action_std)
    dist = Normal(action_mean, 1)
    #dist = Normal(action_mean, action_std)
    action = dist.sample()
    
    log_a = dist.log_prob(action)
    # pylint: disable=E1101 
    #print(policy.policy_history)
    #print(policy.policy_history.size())

    if policy.policy_history.size()[0] != 0:
            # pylint: disable=E1101 
            policy.policy_history = torch.cat([policy.policy_history, log_a])
            # pylint: enable=E1101
    else:
        policy.policy_history = log_a
    
    #torch.cat([policy.policy_history, log_a], dim=1) 
    # pylint: enable=E1101  
    #log_b = log_a.detach().numpy() 
    #policy.policy_history.append(log_b[0])
    
    return action

    """
    def compute_gradient(self, state, action, advantage):
        log_grad = (action - np.dot(state, self.K)) / (self.sigma**2) * state
        grad = advantage * log_grad
        return grad
   
    def gradient_step(self, grad, step_size):
        self.K += step_size * grad

    """
def update_policy(discounted_returns, gamma, policy, optimizer):   
    #zero out gradients
    # pylint: disable=E1101
    # pylint: enable=E110
    optimizer.zero_grad()
    # pylint: disable=E1101 
    discounted_returns = Variable(discounted_returns) 
    #print(policy.policy_history.size()) 
    #print(discounted_returns.size())

    loss = torch.sum(torch.mul(policy.policy_history, discounted_returns.mul(-1)), -1)  
    #print("loss; ", loss) 
    # pylint: disable=E1101
    loss.backward()
    optimizer.step()

    return loss

class LinearValueEstimator(nn.Module):
    def __init__(self, state_dim):
        super(LinearValueEstimator, self).__init__() 

        self.state_dim = state_dim
        self.l1 = nn.Linear(self.state_dim, 20, bias=False) 
        self.l2 = nn.Linear(20, 1, bias = False)

        self.model = torch.nn.Sequential(
            self.l1, 
            nn.Dropout(p=0.6),
            nn.ReLU(), 
            self.l2, 
            nn.Softmax(dim=-1)
        )
 
    def forward(self, state): 
        state = Variable(torch.from_numpy(state)) 
        estimate = self.model(state)
        return estimate 

    def update(self, value_estimate, target, optimizer):
        target = torch.from_numpy(np.array(target))
        loss = nn.MSELoss()
        loss_output = loss(value_estimate, target)
        print(loss_output)
        optimizer.zero_grad()
        loss_output.backward()  
        optimizer.step() 
    
#design linear baseline
"""
class LinearValueEstimator(object):
    def __init__(self, num_states, state): 
        # pylint: disable=E110
        # pylint: enable=E1101
        self.state_dim = len(num_states)
        self.model = ValueEstimateNN(self.state_dim)
        self.value_estimates = self.model(self.state)    

    # TODO: fill this function in
    #takes in a state and predicts a value for the state
    def predict(self): 
        return self.value_estimates

 
    # TODO: fill this function in
    # construct a suitable loss function and use it to update the 
    # values of the value estimator. choose suitable step size for updating the value estimator
"""
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
# a continuous policy 
# a value estimator,
# a discount rate, gammma 
# and the number of episodes you want to run the algorithm for
# make sure to add in the baseline computation here. 
# Using the computed baseline, compute the advantage. 
# Use this advantage in the policy gradient calculation
def reinforce(env, policy,value_estimator, optimizer_p, optimizer_b, gamma, num_episodes):
    
    for episode in range(num_episodes):
        policy.policy_history = Variable(torch.Tensor())
        states = []
        rewards = []
        actions = []
        state = env.reset()
        done = False
        steps = 0
        ep_score = 0 

        while not done:
            action = act(state, policy)
            n_state, reward, done, _ = env.step(action)
            states.append(n_state)
            rewards.append(reward)
            actions.append(action)
            ep_score += reward 
            steps += 1 
            state = n_state

        print("Episode: {}, Score: {} ".format(episode, ep_score))
        episode_length = steps 
    
        discounted_returns = get_discounted_returns(rewards, gamma, episode_length)
        mean_returns = discounted_returns.mean()
        std_returns = np.std(discounted_returns)
        discounted_returns_adj = (discounted_returns - mean_returns) / std_returns 
        disc_ret_ten = torch.FloatTensor(discounted_returns_adj).type(torch.double)

        advantages = []
        # update the policy with SGD a
        """
        for t in range(episode_length):
            estimate = value_estimator(states[t]) 
            estinate = estimate.detach().numpy()
            #estimate_l.append(estinate)
            advantage = discounted_returns_adj[t] - estinate[0] 
            advantages.append(advantage)  
        """
        #value_estimator.update(estimate_l, disc_ret_ten, optimizer_b) 
        advantages_ten = torch.FloatTensor(advantages).type(torch.double)
        update_policy(disc_ret_ten, gamma, policy, optimizer_p) 
        
    #advantages_arr = np.asarray(advantages) 
    #advantages_arr = torch.FloatTensor(advantages_arr).type(torch.double)
           

        

        

if __name__ == "__main__":
    gamma = .999 
    env = Continuous_Pendulum()
    seed = 3
    np.random.seed(seed)
    num_episodes = 2000 
    #define optimizer for policys
    lr_p = 1e-3
    lr_b = .1
    #define optimizer for value estimator
    # TODO: define num_states and num_actions 
    policy = ContinuousPolicy(3, 1, 2).double()
    value_estimator = LinearValueEstimator(3).double()
    # pylint: disable=E1101
    optimizer_p = torch.optim.Adam(policy.parameters(), lr=lr_p) 
    optimizer_b = torch.optim.Adam(value_estimator.parameters(), lr=lr_b)
    # pylint: enable=E1101
    reinforce(env, policy,value_estimator, optimizer_p, optimizer_b, gamma, num_episodes)

    # Test time 
    state = env.reset()
    #env.print_()
    done = False
    state_hist = []
    while not done:
        input("press enter:")
        action = act(state, policy) 
        state, reward, done, _ = env.step(action)
        state_hist.append(state)

    # Plotting test time results
    state_hist = np.array(state_hist)
    plt.plot(state_hist[0, :])
    plt.xlabel("time (s)")
    plt.ylabel("angle (rad)")
    plt.show()
