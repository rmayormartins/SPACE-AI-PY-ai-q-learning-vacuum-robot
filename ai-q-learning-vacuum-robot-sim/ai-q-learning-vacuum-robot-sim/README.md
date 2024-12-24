---
title: ai-q-learning-vacuum-robot-sm
emoji: 🏠⚙️🤖
colorFrom: pink
colorTo: green
sdk: gradio
sdk_version: "4.12.0"
app_file: app.py
pinned: false
---

# AI-Q-Learning-Vacuum-Robot-Cleaner-Simulation

This project is an experimental application v2.0 ...

## Project Overview

This application allows users to train a vacuum robot cleaner to recognize an environment.

## Technical Details

The project utilizes the following technologies:
- **Q-Learning**: Reinforcement learning algorithm for training the robot.
- **Gradio**: Provides an interactive web interface for users to upload images and adjust parameters.

## Instructions

**1- Set up the environment**:
- Edit the grid: 0 = Empty, 1 = Dirt, 2 = Wall, 3 = Vacuum and Generate Environment.
        
**2- Train the robot vacuum cleaner**:
- Reinforcement learning: Q-learning to train the robot vacuum cleaner.
- Start Position Certification: Ensure that the robot does not start in a dirt or wall position.
- Dirt Cleaning: After finding dirt, the robot cleans it, updating the position to 0.
- Reduce Epsilon Decay Rate: This will allow the robot to explore for longer before it starts exploring less.
- Reset the Home State Periodically: To ensure that dirt reappears and the robot has new opportunities to learn.
- Check that the Robot is Not Stuck: A mechanism was add to ensure that the robot is not stuck in a cycle of invalid states.
- Epsilon decay: The decay rate (reduced to 0.999), will allow for more exploration.
- House State Reset: The house is reset every episode to ensure that dirt is present in each new episode.
- Increase the learning rate: Set the alpha to (e.g. 0.2) to see if it helps you learn faster.
- Increase the discount factor: Set the gamma to (e.g. 0.95) to give more value to future rewards.
- Add more randomness to the choice of initial state: This can help to vary training experiences more.
- Reduce the reward when encountering dirt: Reducing the direct reward can make the robot try harder to learn other parts of the environment.
- Add penalties for movement: Adding a small penalty for each movement can encourage the robot to find dirt more efficiently.
- Increase the variation of initial states: Starting from a greater variety of initial positions can help the robot explore more of the environment.
- Change the learning rate (alpha): If you notice that the robot is converging too slowly or too quickly, adjusting the learning rate can help.
- Add more dirt or obstacles: Adding more elements to the environment can make the problem more challenging and interesting for the robot.
- Test different exploration-exploitation (epsilon) policies: Experiment with different epsilon decay strategies to find a good balance between exploration and exploitation.
- Increase the number of episodes: In some cases, training for more episodes can help to further improve the robot's performance.
        
**3- Simulate**:
- New Simulation Grid: 0 = Empty, 1 = Dirt, 2 = Wall, 3 = Vacuum, set iterations (episodes/epochs) and simulate robot.

## License

ECL

## Developer Information

Developed by Ramon Mayor Martins, Ph.D. (2024)
- Email: rmayormartins@gmail.com
- Homepage: [https://rmayormartins.github.io/](https://rmayormartins.github.io/)
- Twitter: @rmayormartins
- GitHub: [https://github.com/rmayormartins](https://github.com/rmayormartins)

## Acknowledgements

Special thanks to Instituto Federal de Santa Catarina (Federal Institute of Santa Catarina) IFSC-São José-Brazil.

## Contact

For any queries or suggestions, please contact the developer using the information provided above.
