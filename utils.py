import numpy as np
import time 

class ReplayBuffer():
    def __init__(self, length = 10000):
        # Buffer Collection: (S, A, R, S', D)
        # Done represents a mask of either 0 and 1
        self.length = length
        self.buffer = []

    def add(self, sample):
        if (len(self.buffer) > self.length):
            self.buffer.pop(0)
        self.buffer.append(sample)

    def sample(self, batch_size):
        idx = np.random.permutation(len(self.buffer))[:batch_size]
        state_b = []
        action_b = []
        reward_b = []
        nextstate_b = []
        done_b = []
        for i in idx:
            s, a, r, sp, d = self.buffer[i]
            state_b.append(s)
            action_b.append(a)
            reward_b.append(r)
            nextstate_b.append(sp)
            done_b.append(d)
        state_b = np.array(state_b)
        action_b = np.array(action_b)
        reward_b = np.array(reward_b)
        nextstate_b = np.array(nextstate_b)
        done_b = np.array(done_b)
        return (state_b, action_b, reward_b, nextstate_b, done_b)


def print_eta(start_time, i, epIter, last_reward):
    time_passed = (time.time() - start_time) / 3600
    minutes = (time_passed - int(time_passed)) * 60
    estimated_time = "{:02.0f}:{:02.0f}".format(int(time_passed), int(minutes))

    time_left = (epIter - i - 1) * (time_passed / (i + 1))
    minutes = (time_left - int(time_left)) * 60
    estimated_time += " (Estimated remaining time: {:02.0f}:{:02.0f})".format(int(time_left), int(minutes))

    print("Episode: {}/{} | Last Reward: {} | Runtime (hours): {}  \t\t\t\t\t\t\t\t\t\r".format(i + 1, epIter, round(last_reward,2), estimated_time), end='')

def baseLineEp(env, STEPS_PER_EPISODE):
    state = env.reset()[0]
    r = 0
    running_step = 0
    while(True):
        action = np.random.normal(size = env.action_space.shape[0])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        done = done or running_step >= STEPS_PER_EPISODE
        running_step += 1
        r += reward
        if done:
            print(" > Random: Reward at Termination: {}".format(r))
            return
