# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
#
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
#
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
#
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C  # Sample RL Algo imports
from sb3_contrib import RecurrentPPO  # Importing an LSTM


# To run the sample TTNN model, you can uncomment the 2 lines below:
# import ttnn
# from user.my_agent_tt import TTMLPPolicy


class SubmittedAgent(Agent):
    """
    Input the **file_path** to your agent here for submission!
    """

    def __init__(self, file_path: Optional[str] = None):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = True
        self.skills = {}
        self.active_skill = None
        self.time = 0
        self.prevpos = [0, 0]
        self.lstm_states = None
        self.episode_starts = True

    def _initialize(self) -> None:
        self.file_path = self._gdown()
        self.model = RecurrentPPO.load(self.file_path)

    def _gdown(self) -> str:
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1BrD5DL3xe_LGq27bRnjftlXOqkB6gO6H/view?usp=share_link"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def reset(self) -> None:
        self.episode_starts = True

    def press(self, action, keys):
        # For pressing keys
        key_mask = self.act_helper.press_keys(keys)
        result = [max(1 if k > 0 else 0, a) for a, k in zip(action, key_mask)]
        return result

    def _apply_mask(self, action, keys, zero_out=True):
        key_mask = self.act_helper.press_keys(keys)
        result = []
        for a, k in zip(action, key_mask):
            if k > 0 and zero_out:
                result.append(0)
            else:
                result.append(a)
        return result

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, "player_pos")
        weapon = self.obs_helper.get_section(obs, "player_weapon_type")
        player_jumps_left = self.obs_helper.get_section(obs, "player_jumps_left")
        opp_pos = self.obs_helper.get_section(obs, "opponent_pos")
        opp_KO = self.obs_helper.get_section(obs, "opponent_state") in [5, 11]
        action = self.act_helper.zeros()

        # Attack if near
        if (
            ((pos[0] - opp_pos[0]) ** 2 + (pos[1] - opp_pos[1]) ** 2 < 4.0)
            and not opp_KO
            and pos[0] > -7
            and pos[0] < 7
        ):

            action, self.lstm_states = self.model.predict(
                obs,
                state=self.lstm_states,
                episode_start=self.episode_starts,
                deterministic=False,
            )
            self.episode_starts = False

            # Block "g"
            action = action = self._apply_mask(action, ["g"])

            # Block "h"
            if weapon != [0.0]:
                action = self._apply_mask(action, ["h"])
            else:
                if self.time % 2 == 0:
                    action = self.press(action, ["h"])
        else:
            # Pick up weapon
            if weapon == [0.0] and self.time % 2 == 0:
                action = self.press(action, ["h"])

            if pos[0] > 10.67 / 2:
                action = self.press(action, ["a"])
            elif pos[0] < -10.67 / 2:
                action = self.press(action, ["d"])
            elif not opp_KO and opp_pos[1] < 3:
                # Head toward opponent
                if opp_pos[0] > pos[0]:
                    action = self.press(action, ["d"])
                else:
                    action = self.press(action, ["a"])
            else:
                action = self.press(action, ["a"])

            # Note: Passing in partial action
            # Jump if below map or opponent is above you
            if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
                action = self.press(action, ["space"])

        if self.episode_starts:
            self.episode_starts = False

        if pos[0] > 7 or pos[0] < -7 or (pos[0] > -2 and pos[0] < 2):
            action = self._apply_mask(action, ["s"])

        if (pos[0] > -2 and pos[0] < 2) and player_jumps_left == 0:
            action = self._apply_mask(action, ["d"])
            action = self.press(action, ["a"])
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 5, verbose=2):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
