import vizdoom

import matplotlib.pyplot as plt

import numpy as np

import cv2

import time

from utils import *

def create_environment():
    env = vizdoom.DoomGame()
    env.load_config("deadly_corridor.cfg")
    env.set_mode(vizdoom.Mode.SPECTATOR)
    env.init()

    return env

def preprocess_frame(frame):
    whole_rgb_img = (frame).astype(np.uint8)[:-35]
    resized_img = Image.fromarray(whole_rgb_img, "RGB").resize((64, 64))
    greyscaled_img = resized_img.convert("L")
    scaled_img = np.array(greyscaled_img) / 255.
    print(np.array(scaled_img).shape)

    return scaled_img

class return_arrays():
    def __init__(self, i):
        self.i = i
        self.mem_size = 2500
        self.states = np.zeros((self.mem_size, *(64, 64, 4)), dtype=np.float32)
        self.actions = np.zeros((self.mem_size, *(7,)), dtype=np.uint8)
        self.rewards = np.zeros(self.mem_size, dtype=np.float32)
        self.n_states = np.zeros((self.mem_size, *(64, 64, 4)), dtype=np.float32)
        self.dones = np.zeros(self.mem_size, dtype=np.bool)
        
    def add(self, index, s, a, r, ns, d):
        
        self.states[index] = s
        self.actions[index] = a
        self.rewards[index] = r
        self.n_states[index] = ns
        self.dones[index] = d

    def save(self, last_step):
        self.states, self.actions, self.rewards, self.n_states, self.dones = self.states[:last_step+1], self.actions[:last_step+1], self.rewards[:last_step+1], self.n_states[:last_step+1], self.dones[:last_step+1]
        np.save("vizdoom_dataset/states{}.npy".format(self.i), self.states)
        np.save("vizdoom_dataset/actions{}.npy".format(self.i), self.actions)
        np.save("vizdoom_dataset/rewards{}.npy".format(self.i), self.rewards)
        np.save("vizdoom_dataset/n_states{}.npy".format(self.i), self.n_states)
        np.save("vizdoom_dataset/dones{}.npy".format(self.i), self.dones)

env = create_environment()
episodes = 10

stack_size = 4
stacked_frames = deque([np.zeros((64, 64), dtype=np.uint8) for i in range(stack_size)], maxlen=4) 

for i in range(episodes):
    print("Episode: ", i+1)

    stacked_frames = deque([np.zeros((64, 64), dtype=np.uint8) for i in range(stack_size)], maxlen=4) 

    video = return_arrays(i)
    frames = []
    last_step = 0

    env.new_episode()

    s = env.get_state()
    s = np.stack(s.screen_buffer, axis=-1)
    s = preprocess_frame(s)
    s, stacked_frames = stack_frames(stacked_frames, s, True)
    for step in range(2500):
        
        env.advance_action()
        a = env.get_last_action()
        r = env.get_last_reward()
        d = env.is_episode_finished()

        if d:
            ns = np.zeros((64, 64), dtype=np.int)
            ns, stacked_frames = stack_frames(stacked_frames, ns, False)
            video.add(step, s, a, r, ns, d)
            break

        else:
            ns = np.stack(env.get_state().screen_buffer, axis=-1)
            ns = preprocess_frame(ns)
            ns, stacked_frames = stack_frames(stacked_frames, ns, False)
            video.add(step, s, a, r, ns, d)
            s = ns

        last_step = step
        
    video.save(last_step)
    print("...saved...")
    '''frames.append(cv2.cvtColor(np.array(s).astype(np.uint8), cv2.COLOR_BGR2RGB))
    video.save(last_step)
    print("...saved...")

    size = (320, 240)
    out = cv2.VideoWriter("videos//video{}.mp4".format(time.time()), 0x7634706d, 20, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()'''
    
