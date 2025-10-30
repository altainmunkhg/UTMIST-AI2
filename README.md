# UTMIST AI^2 2025

[Colab Notebook](https://colab.research.google.com/drive/1V184vtHSagN13L0SbWGmnY-jCDvIefmm?usp=sharing) | [Paper](https://drive.google.com/file/d/1G0hatGPBXvh2j5byjfrqKBthknNOt5sp/view)

This is the official repository of the source code of AI Squared (AI^2) -- a full multi-agent 1v1 RL environment and tournament server, written for the [University of Toronto Machine Intelligence Student Team (UTMIST)](https://utmist.gitlab.io/) 2025 AI^2 Tournament.

### Project Description

The UTMIST AI^2 Tournament is UTMIST's flagship hackathon/tournament event focused on reinforcement learning launched in February 2025 to 400+ participants. The purpose of this project is to expose and allow participants to apply reinforcement learning in a fun and engaging game environment. The game environment was programmed in Pygame and PyMunk and was inspired by platform-fighting games such as Brawlhalla, with a custom multi-agent RL framework built with SB3 that allows participants to design, train, and battle their AI agents in a 1v1 fighting game. The AI^2 website and server allows for agent submissions and running multiplayer tournament matches between participants. Through this tournament, we aim not only to celebrate the creativity and technical skills of our participants, but also to continue building a dynamic and inclusive platform that fosters deeper engagement with reinforcement learning.

### Tournament Match Demo

<img src="docs/aisquaredv4.gif"></img>

### On mps

Add the fallback using `PYTORCH_ENABLE_MPS_FALLBACK=1` when running anything using mps.
