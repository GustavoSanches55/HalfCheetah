import gym
from DDPG import DDPG
from utils import ReplayBuffer, print_eta, baseLineEp
import torch
import numpy as np
import argparse 
import time 
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='DDPG_cheetah', help="Name of the model to save")
parser.add_argument("--render", default=False, action='store_true', help="Render the environment during training")
parser.add_argument("--episodes", default=200, type=int, help="Number of episodes to train for")
parser.add_argument("--steps", default=1000, type=int, help="Number of steps per episode")
parser.add_argument("--batch", default=128, type=int, help="Batch size for training")
args = parser.parse_args()

MODEL_NAME = args.model
REALTIME_RENDER = args.render
NUM_EPISODES = args.episodes
STEPS_PER_EPISODE = args.steps
BATCH_SIZE = args.batch
RENDER_VIDEO = True
start_time = time.time()

def recordEps(env, model, save_path=None):
    if not REALTIME_RENDER and RENDER_VIDEO:
        rec = gym.wrappers.monitoring.video_recorder.VideoRecorder(env, save_path)
    state = env.reset()[0]
    r = 0
    step = 0
    while(True):
        action = model.predict(np.expand_dims(state, axis=0))
        if not REALTIME_RENDER and RENDER_VIDEO:
            rec.capture_frame()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated or step >= STEPS_PER_EPISODE
        r += reward
        step += 1
        if done:
            if not REALTIME_RENDER and RENDER_VIDEO:
                rec.close()
            print("Reward at termination: {}".format(r))
            print("Avg Reward: {}".format(r/step))
            return

def runEp(env, memory, model, returnReward=False):
    # function to run an episode and train the model
    step = 0
    epochReward = []
    timestamps = []
    stepList = []
    epReward = 0
    state = env.reset()[0]
    while True:
        action = model.predict(np.expand_dims(state, axis=0))
        
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated or step >= STEPS_PER_EPISODE
        
        #  Add transition tuple to replay buffer
        memory.add((state, action, reward, new_state, done))
        state = new_state
        step += 1
        
        #  Train every 50 steps
        if (step%50) == 0:
            for _ in range(50):
                model.train_step(memory, BATCH_SIZE) 
        if returnReward:
            epReward += reward
        if done:
            state = env.reset()[0]
            done = False
            if returnReward:
                stepList.append(step)
                epochReward.append(epReward)
                timestamps.append(time.time() - start_time)
                epReward = 0
            break
    if returnReward:
        return stepList, epochReward, action, timestamps
    return step

def train_model(env, memory, model, epIter=200):
    reward_plot = [] # Format [(step, reward, timestamp)]
    start_time = time.time()
    tStep = 0
    epochReward = [0]
    for i in range(epIter):
        if ((i+1)%3 == 0): # Every 3 episodes, run an episode and return the reward plot
            stepList, epochReward, act, timestamps = runEp(env, memory, model, returnReward=True)
            step = stepList[-1]
            for j in range(len(stepList)):
                stepList[j] += tStep
            reward_plot.extend([[a,b,c] for a,b,c in zip(stepList, epochReward, timestamps)])
        else:
            step = runEp(env, memory, model) # Run episode and train model
        if ((i+1)%5 == 0): # Save and render every 5 episodes
            time_passed = (time.time() - start_time) / 3600
            minutes = (time_passed - int(time_passed)) * 60
            estimated_time = "{:02.0f}:{:02.0f}".format(int(time_passed), int(minutes))
            print("Last Reward: {} Runtime (hours) at checkpoint: {}                                             ".format(round(epochReward[-1],2), estimated_time))
            
            # saving the reward plot as a csv file
            df = pd.DataFrame(reward_plot)
            df.to_csv(f'results/{MODEL_NAME}.csv', index=False)
            
            model.save(path = f'models/{MODEL_NAME}')
        if REALTIME_RENDER:
                env.render()
        tStep += step
        
        print_eta(start_time, i, epIter, epochReward[-1])

def main():
    print("Model name (--model): {}".format(MODEL_NAME))
    print("Realtime render (--render flag): {}".format(REALTIME_RENDER))
    start_time = time.time()
    envName = 'HalfCheetah-v2'
    print("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("Device Count: {}".format(torch.cuda.device(0)))
        print("Device Name (first): {}".format(torch.cuda.get_device_name(0)))
    if REALTIME_RENDER:
        env = gym.make(envName, render_mode="human")
    else:
        env = gym.make(envName, render_mode="rgb_array")
    action_space = env.action_space.shape[0]
    state_space = env.observation_space.shape[0]
    
    model = DDPG(state_space, action_space)
    rmemory = ReplayBuffer(int(1e6))
    print("Starting train...")
    train_model(env, rmemory, model, epIter=NUM_EPISODES)
    print("\nStarting evaluation...")

    recordEps(env, model, save_path=f'vid/{MODEL_NAME}.mp4')

    print("Baseline for comparison:")
    for _ in range(5):
        baseLineEp(env, STEPS_PER_EPISODE)
    time_passed = (time.time() - start_time) / 3600
    minutes = (time_passed - int(time_passed)) * 60
    time_passed = "{:02.0f}:{:02.0f}".format(int(time_passed), int(minutes))
    print("Total Runtime (hours): {}   \t\t\t\t\t\t\t\t\t ".format(time_passed))

    if REALTIME_RENDER:
        env.reset()
        env.close()

if __name__ == '__main__':
    main()

