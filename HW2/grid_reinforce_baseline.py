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

    def softmax(self, x):
        ex = np.exp(x)
        sum_ex = np.sum(np.exp(x))
        return ex/sum_ex

    def get_pi(self, state):
        logits = self.weights[state, : ]
        pi = self.softmax(logits)
        return pi
    
    def act(self, state):
        pi = self.get_pi(state)
        action = np.random.choice(self.num_actions, p=pi)
        return action

    # TODO: fill this function in    
    # computes the gradient of the discounted return    
    # at a specific state and action    
    # use the computed advantage function appropriately.
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, advantage):
        pi = self.get_pi(state)

        #make onehot array representing current action
        act_onehot = np.zeros(pi.shape)
        act_onehot[action, ...] = 1.0

        #calculate grad(logpi)
        dsoftmax = pi * (act_onehot - pi)
        dlog = dsoftmax/pi
        w_grad = np.zeros((self.num_states, self.num_actions))
        w_grad[state, :] = dlog * advantage
        return w_grad



    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())    
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.weights += step_size * grad

class ValueEstimator(object):
    def __init__(self, num_states):
        self.num_states = num_states
        self.values = np.zeros((self.num_states))

    def predict(self,state):
        return self.values[state]

    def update(self, state, value_estimate, target, value_step_size):
        l1_loss = np.abs(target - value_estimate)
        self.values[state] += value_step_size * l1_loss

# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]

def get_discounted_returns(rewards, gamma):
    discounted_returns = []
    powers = np.arange(0, len(rewards))
    gammas = np.ones((powers.shape)) * gamma
    gamma_poly = np.power(gammas, powers)
    for i in range(len(rewards)):
        future_rewards = rewards[i:]
        future_gammas = []
        if(i > 0):
            future_gammas = gamma_poly[:-i]
        else:
            future_gammas = gamma_poly
        discounted_returns.append(np.sum(np.multiply(future_gammas, future_rewards)))
    return discounted_returns

# TODO: fill this function in 
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a value estimator,
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
# make sure to add in the baseline computation here. 
# Using the computed baseline, compute the advantage. 
# Use this advantage in the policy gradient calculation
def reinforce(env, policy, gamma, num_episodes, learning_rate, value_estimator, vs_scale = 1e-2):
    total = []
    for episode in range(num_episodes):
        rewards = []
        states = []
        actions = []
        ep_reward = 0
        episode_length = 0
        state = env.reset()
        while(True):
            action = policy.act(state)
            n_state, reward, done = env.step(action)
            states.append(state)
            rewards.append(reward)
            ep_reward += reward
            actions.append(action)
            episode_length += 1
            state = n_state
            if(done):
                break
        total.append(ep_reward)
        if(episode % 4 == 0):
            print(ep_reward)
        discounted_returns = get_discounted_returns(rewards, gamma)
        grad_v = np.zeros((policy.weights.shape))
        for step in range(episode_length):
            s = states[step]
            a = actions[step]
            disc_ret = discounted_returns[step]
            advantage = disc_ret - value_estimator.predict(s)
            grad_v += policy.compute_gradient(s, a, advantage)
            value_estimator.update(s, value_estimator.predict(s), disc_ret, learning_rate * vs_scale)
        policy.gradient_step(grad_v, learning_rate)
    return total

if __name__ == "__main__":
    np.random.seed(6)
    env = GridWorld(MAP1)
    env.print()

    gamma = 0.9
    num_episodes = 20000
    learning_rate = 1e-3
    env = GridWorld(MAP2)

    #temperature must be tuned.
    policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions())
    value_estimator = ValueEstimator(env.get_num_states())
    reinforce(env, policy, gamma, num_episodes, learning_rate, value_estimator)

    #Test time
    state = env.reset()
    env.print()
    done = False
    while not done:
        input("press enter:")
        action = policy.act(state)
        state, reward, done = env.step(action)
        env.print()