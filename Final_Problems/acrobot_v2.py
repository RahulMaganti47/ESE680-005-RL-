import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Categorical
import gym
from collections import deque
import numpy as np 
from torch.distributions import Categorical
import matplotlib.pyplot as plt


class Policy(nn.Module): 
    def __init__(self, s_size=6, h_size=50, a_size=3):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(s_size, h_size)
        self.l2 = nn.Linear(h_size, a_size)

        self.model = nn.Sequential(
            self.l1, 
            nn.ReLU(), 
            self.l2, 
            nn.Softmax(dim=1)
        )
         
    def forward(self, x):
        return self.model(x)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item() - 1, m.log_prob(action)

def compute_rewards(rewards, gamma):
    discounted_rewards = np.zeros(len(rewards))
    moving_add = 0
    for i in reversed(range(0, len(rewards))):
        moving_add = moving_add*gamma + rewards[i]
        discounted_rewards[i] = moving_add

    return discounted_rewards

class Critic(nn.Module): 
    def __init__(self, state_dim=6, hidden_dim=20, output_dim=1, lambd=.9):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.lambd = lambd

    def forward(self, state): 
        state = torch.from_numpy(state).float().unsqueeze(0) 
        x = F.relu(self.l1(state)) 
        x = self.l2(x) 
        return F.softmax(x, dim=1)

    def td_error(self, reward, value_next, value_now, gamma, done, I): 
        if done: I = I * gamma 
        td_error = reward + gamma*(1-done)*value_next - value_now 
        return td_error 
        
def train(n_episodes, policy, critic, gamma, print_every=4):
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    optimizer_v = optim.Adam(critic.parameters(), lr=.001)
    scores_deque = deque(maxlen=100)
    
    total_rewards = []
    for ep in range(n_episodes): 
        traj_log_probs = []
        rewards = []
        state = env.reset()
        score = 0 
        I = 1.0 
        done = False
 
        while not done:
            action, log_prob = policy.act(state)
            #value_func = critic.forward(state)
            traj_log_probs.append(log_prob)
            next_state, reward, done, _ = env.step(action)
            #value_func_next = critic(next_state)
            #td_error = critic.td_error(reward, value_func_next, value_func, gamma, done, I) 
            
            score += reward
            rewards.append(reward)
           
        scores_deque.append(score) 
        total_rewards.append(score)
                
        disc_rewards = compute_rewards(rewards, gamma)
        disc_rewards = torch.tensor(disc_rewards)
        
        policy_loss = [] 
        for t, log_prob in enumerate(traj_log_probs):
            policy_loss.append(-log_prob * disc_rewards[t])  
        policy_loss = torch.cat(policy_loss).sum() 
        
        #value_loss = F.l1_loss(value, torch.tensor([disc_rewards]))
        #add gradient trace 
        #for p in critic.parameters(): 
        #    p.grad = p.grad * critic.lambd
        
        #loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # backprop
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if ep % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(ep, np.mean(scores_deque)))        
    
    return total_rewards

if __name__ == "__main__":

    env = gym.make('Acrobot-v1')
    print('observation space:', env.observation_space)
    print('action space:', env.action_space)

    num_episodes = 100 
    gamma = .99
    policy = Policy()
    critic = Critic() 
    total_rewards = train(num_episodes, policy, critic, gamma)
    
    #num_episodes = np.linspace(1, num_episodes, 4)
    #assert (len(num_episodes) == len(total_rewards)) 
    
    # Plot 
    plt.plot(range(len(total_rewards)), total_rewards, 'b-')
    plt.title('Episodes vs. Cumulative Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    pl.show()
