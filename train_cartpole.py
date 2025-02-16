from agent import Cartpole_RL_Agent
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

def make_video(agent):
    env = gym.make('CartPole-v1', render_mode="rgb_array")

    # RecordVideo is convenient cause it will simply auto-save the video!
    env = RecordVideo(env, video_folder="video-demo", name_prefix="eval",
                  episode_trigger=lambda x: True)
    done = False
    state, _ = env.reset() # split out the tuple, trash second info

    while not done:
        state_qn = np.expand_dims(state, axis=0)
        q_values = agent.q_network(state_qn)
        action = agent.get_action(q_values)
        state, _, done, terminated, _ = env.step(action)
        done = done or terminated

    env.close()

env = gym.make("CartPole-v1")
state_size = env.observation_space.shape
action_size = env.action_space.n # only 2 actions left or right!

# Number of training episodes
num_episodes = 800

# Define the agent
agent = Cartpole_RL_Agent(state_size, action_size)

option = input("Provide your choice: ")
if option == "1":
    agent.train_episodes(num_episodes)
else:
    agent.load_model()
make_video(agent)