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
    BasedRunningAgent,
)  # add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame

pygame.init()

my_agent = UserInputAgent()
#my_agent = SubmittedAgent(file_path="checkpoints/Hierarch_Experiment_3.1_Combat/rl_model_4003419_steps.zip")
#my_agent = BasedAgent()
my_agent = CustomAgent(file_path='checkpoints/Custom_Experiment_1/rl_model_8237700_steps')

# Input your file path here in SubmittedAgent if you are loading a model:
#opponent = SubmittedAgent(file_path='checkpoints/Hierarch_Experiment_4_combat/rl_model_18954000_steps.zip')
#opponent = BasedAgent()
opponent = CustomAgent(file_path='checkpoints/Custom_Experiment_1/rl_model_7233300_steps')

match_time = 999999

# Run a single real-time match
run_real_time_match(
    agent_1=my_agent, #orange
    agent_2=opponent,  #blue
    max_timesteps=30 * 999990000,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)
