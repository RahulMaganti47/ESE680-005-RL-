from mountain_car import *
import numpy as np
import pdb
from scipy.stats import truncnorm 

class LinearPolicy(object):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        # here are the weights for the policy - you may change this initialization
        # K[0] and K[1] represent the position and velocity
        self.K = np.zeros(2)
        self.sigma = .3
        
    def get_truncated_normal(self, mean, sd, low, upp):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    # TODO: fill this function in 
    # it should take in an environment state
    # return the action that follows the policy's distribution
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
    
    def act_norm(self,state):
        mu = np.dot(self.K, state)
        action = np.random.normal(mu, self.sigma, 1)
        #the mountain_ca.py already clips the values, so no point in duplicating the effort
        return action

    def act_trunc(self, state):
        mu = np.dot(self.K, state) 
        probs_a = self.get_truncated_normal(mu, self.sigma, -1, 1)
        action = probs_a.rvs()
        act_a = np.array([action])
        return act_a

    # TODO: fill this function in
    # computes the gradient of the discounted return 
    # at a specific state and action
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, states, actions, discounted_returns):
        '''
        gradient of the linear controller with respect to K takes the form
        (action*state - K*state^2)/sigma
        '''
        '''
        grad = np.zeros(state.shape)
        mean = np.dot(state, self.K).reshape(action.shape)
        d_sa = np.zeros(state.shape, dtype='float64')
        d_sa[:,0] = (action - mean) * state[:,0].reshape(action.shape)
        d_sa[:,1] = (action - mean) * state[:,1] 
        d_scaled = d_sa /(self.sigma**2) 
        #pdb.set_trace()
        grad = d_scaled*discounted_return
        return grad
        '''
        '''
        Nikil's potentially correct vectorized version
        (t,1)*(t,2) - (1,2)*(t,2) = (t,2)
        (actions*states - K*states^2)/sigma
        (t,2)*(t,1) = (t,2)
        sum along axis 1 = (1,2)
        '''
        gradients = (actions*states + self.K*(states**2) )/self.sigma
        return np.sum(gradients*discounted_returns,axis=0)



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
def get_discounted_returns(rewardsList, gamma):
    '''
    #Rahul's version
    moving_add = 0 
    discounted_rewards = np.zeros(time_steps)
    for i in reversed(range(0, len(rewards))):
        moving_add = gamma*moving_add + rewards[i]
        discounted_rewards[i] = moving_add
    
    return discounted_rewards

    '''
    #Nikil's vectorized version
    rewards = np.array(rewardsList)
    rewards = rewards.reshape(1,rewards.size)
    to_the_nth = np.arange(rewards.size).reshape(rewards.shape)
    grid = (to_the_nth) + (-1*to_the_nth.transpose())
    gamma_grid = np.power(gamma,grid)
    gamma_grid[grid<0]= 0
    return np.sum(rewards*gamma_grid,axis=1).reshape(-1,1)

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

        #generate trajectory for this episode
        while not done:

            a = policy.act_norm(state)
            next_state, reward, done, _ = env.step(a)
            rewards.append(reward)
            states.append(next_state)
            actions.append(a)
            score += reward 
            state = next_state

            #prevent over-long, never-ending trials
            if t==999:
                done = True
            
            t += 1
        num_time_steps = t
        print("score: ", score,"\n")

        #learn from this episode
        discounted_returns = get_discounted_returns(rewards, gamma) 
        grad = np.zeros(policy.K.shape)
        grad += policy.compute_gradient(np.array(states), np.array(actions), discounted_returns)
        policy.gradient_step(grad, learning_rate)
         
    return policy.K 


if __name__ == "__main__": 
    gamma = 0.9
    num_episodes = 10000
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
