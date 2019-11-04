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
    def __init__(self, num_states, num_actions):
        self.num_states = num_states 
        self.num_actions = num_actions
        
        #
        self.layer1 = nn.Linear(self.num_states, 20, bias=False)
        self.action_mu = nn.Linear(20, self.num_actions, bias=False) 

        self.action_sigma = nn.Linear(20, self.num_actions, bias=False)

    
    def forward(self, x):

        #forward pass for mean
        model = torch.nn.Sequential( 
            self.layer1,
            nn.Dropout(p=0.6),
              nn.ReLU(),
            self.action_mu,
            nn.Softmax(dim=-1)  
        ) 

        #forward pass for sigma
        model2 = torch.nn.Sequential(
            self.layer1, 
            nn.Dropout(p=0.6),
                nn.ReLU(), 
            self.action_sigma, 
            nn.Softmax(dim=-1) 
        )

        return model(x), model2(x)


    # TODO: fill this function in  
    # it should take in an environment state
def act(self, state):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    # pylint: disable=E1101
    state = torch.from_numpy(state).type(torch.double)
    # pylint: enable=E1101
    action_mean, action_std = policy(Variable(state))
    #dist = Normal(action_mean, action_std)
    dist = Normal(action_mean, action_std)
    #dist = Normal(action_mean, action_std)
    action = dist.sample()

    # TODO: Move this to its own function
    # Add log probability of our chosen action to our history    
    if policy.policy_history.size()[0] != 0:
            # pylint: disable=E1101
            policy.policy_history = torch.cat([policy.policy_history, dist.log_prob(action)])
            # pylint: enable=E1101

def update_policy(discounted_returns, gamma, policy, optimizer):   
    # pylint: disable=E1101
    loss = torch.sum(torch.mul(policy.policy_history, Variable(discounted_returns).t().mul(-1)), -1)   
    # pylint: disable=E1101
  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss
"""
#class RBFnet(nn.Module): 
    def __init__(self, centers, num_states):
        self.centers = nn.Paramter(centers)
        self.beta = nn.Parameter(torch.ones(1,self.num_centers)/10)
        self.linear = nn.Linear(self.num_centers, self.num_states, bias = False) 
        self.num_centers = num_states
    
    def kernel_func(self, i_data): 
        input_n = self.num_states
        # reshaping dimensions 
        features = torch.exp(-self.beta.mul((i_data-self.centers).pow(2).sum(2,keepdim=False).sqrt()))
        #features_vec = np.zeros(self.num_states)
        #features_vec = exp(-self.beta * norm(i_data-centers)**2) 
        return features 

    def forward(self, i_data):
        rbf = self.kernel_func(i_data)
        values = self.linear(rbf)
        return values 
 """

class ValueEstimateNN(nn.Module):
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.l1 = nn.Linear(self.state_dim, 20, bias=False) 
        self.l2 = nn.Linear(20, 1, bias = False)

    def forward(self, x): 
        h_relu = F.relu(self.l1(x))
        estimate  = self.l2(h_relu)
        return estimate 

#design linear baseline
class LinearValueEstimator(object):
    def __init__(self, num_states, x):
        self.num_states = num_states
        # pylint: disable=E1101
        self.x = Variable(torch.randn(1, self.num_states))
        # pylint: enable=E1101
        self.model = ValueEstimateNN(self.num_states)
        self.value_estimates = self.model(self.x)    

    # TODO: fill this function in
    #takes in a state and predicts a value for the state
    def predict(self, state): 
        return self.outputs.numpy() 
    
    # TODO: fill this function in
    # construct a suitable loss function and use it to update the 
    # values of the value estimator. choose suitable step size for updating the value estimator
    def update(self, state, value_estimates, target, optimizer):
        loss = nn.MSELoss()
        loss_output = loss(value_estimates, target)
        optimizer.zero_grad()
        loss_output.backward() 
        optimizer.step() 

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
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
# make sure to add in the baseline computation here. 
# Using the computed baseline, compute the advantage. 
# Use this advantage in the policy gradient calculation
def reinforce(env, policy,value_estimator, optimizer_p, optimizer_b, gamma, num_episodes):
  
    for episode in range(num_episodes):
        states = []
        rewards = []
        actions = []
        state = env.reset()
        done = False
        steps = 0
        ep_score = 0 
        
        while not done: 
            action = policy.act(state)
            n_state, reward, done = env.step(action)
            states.append(n_state)
            rewards.append(reward)
            actions.append(action)
            ep_score += reward 
            steps += 1 
            state = n_state

        print("Episode: {}, Score: {}".format(episode, ep_score))
        episode_length = steps 
        disc_ret = get_discounted_returns(rewards, gamma, episode_length)
        
        # update the policy with SGD 
        update_policy(disc_ret, gamma, policy, optimizer)
        value_estimator.update(s)
        

if __name__ == "__main__":
    env = Continuous_Pendulum()
    #define optimizer for policys
    lr_p = .01
    lr_b = .01 
    optimizer_policy = optim.SGD(policy.parameters(), lr=lr_p)
    #define optimizer for value estimator
    optimizer_val = torch.optim.SGD(model.parameters(), lr = lr_b, momentum=0.9)
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
