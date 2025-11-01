from environment.environment import RenderMode, CameraResolution
from environment.agent import run_real_time_match
from user.train_agent import (
    UserInputAgent,
    BasedAgent,
    ConstantAgent,
    ClockworkAgent,
    SB3Agent,
    RecurrentPPOAgent,
    CustomAgent,
)  # add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame

pygame.init()

steps = 30_003

my_agent = UserInputAgent()
# my_agent = BasedAgent()

# Input your file path here in SubmittedAgent if you are loading a model:
# opponent = SubmittedAgent(file_path='checkpoints/Hierarch_Experiment_2_Movement/rl_model_12142434_steps')
opponent = SubmittedAgent(file_path=f"checkpoints/SB3_PPO_3/rl_model_{steps}_steps")
# opponent = CustomAgent()

match_time = 999999

# Run a single real-time match
run_real_time_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * 999990000,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)
