import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time
import imageio
import os

# 
grid_size = (10, 10)
house = np.zeros(grid_size, dtype=int)
vacuum_pos = None
q_table = np.zeros((grid_size[0], grid_size[1], 4))
initial_house = None

def draw_grid(house, vacuum_pos, iteration=None):
    fig, ax = plt.subplots()
    block_size = 1
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            if house[x, y] == 1:
                rect = patches.Rectangle((y, x), block_size, block_size, linewidth=1, edgecolor='black', facecolor='green')
            elif house[x, y] == 2:
                rect = patches.Rectangle((y, x), block_size, block_size, linewidth=1, edgecolor='black', facecolor='red')
            elif house[x, y] == 3:
                rect = patches.Rectangle((y, x), block_size, block_size, linewidth=1, edgecolor='black', facecolor='blue')
            else:
                rect = patches.Rectangle((y, x), block_size, block_size, linewidth=1, edgecolor='black', facecolor='white')
            ax.add_patch(rect)
    if vacuum_pos:
        robot = patches.Circle((vacuum_pos[1] + 0.5, vacuum_pos[0] + 0.5), 0.3, linewidth=1, edgecolor='blue', facecolor='blue')
        ax.add_patch(robot)
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.title(f'Environment Configuration' if iteration is None else f'Iteration: {iteration}')
    plt.legend(handles=[
        patches.Patch(color='green', label='Dirt'),
        patches.Patch(color='red', label='Wall'),
        patches.Patch(color='white', label='Empty'),
        patches.Patch(color='blue', label='Vacuum')
    ], bbox_to_anchor=(1.05, 1), loc='upper left')
    if iteration is not None:
        plt.savefig(f"iteration_{iteration}.png")
    else:
        plt.savefig("grid.png")
    plt.close()
    return f"iteration_{iteration}.png" if iteration is not None else "grid.png"

def update_grid(grid):
    global house, vacuum_pos, initial_house
    house = np.array(grid, dtype=int)
    initial_house = house.copy()  #
    vacuum_pos = tuple(np.argwhere(house == 3)[0]) if 3 in house else None
    return draw_grid(house, vacuum_pos)

def reset_house():
    global house
    house = initial_house.copy()  #

def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, 3)  # Explore
    else:
        action = np.argmax(q_table[state[0], state[1]])  # Exploit
    return action

def get_next_state(state, action):
    if action == 0:  # up
        next_state = (max(state[0] - 1, 0), state[1])
    elif action == 1:  # down
        next_state = (min(state[0] + 1, grid_size[0] - 1), state[1])
    elif action == 2:  # left
        next_state = (state[0], max(state[1] - 1, 0))
    else:  # right
        next_state = (state[0], min(state[1] + 1, grid_size[1] - 1))
    return next_state

def is_valid_state(state):
    return house[state] != -1

def train_robot(episodes, alpha, gamma, epsilon, epsilon_decay, epsilon_min, move_penalty):
    global house, vacuum_pos, q_table, max_steps_per_episode
    rewards_per_episode = []
    max_steps_per_episode = 200  #
    episode_log = []

    for episode in range(episodes):
        reset_house()  # 
        state = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
        while not is_valid_state(state) or house[state] == 1:  # 
            state = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))

        steps = 0
        total_reward = 0
        episode_info = []

        while steps < max_steps_per_episode:
            action = choose_action(state, epsilon)
            next_state = get_next_state(state, action)
            if is_valid_state(next_state):
                reward = 1 if house[next_state] == 1 else move_penalty  # 
                q_table[state[0], state[1], action] = q_table[state[0], state[1], action] + \
                    alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1]]) - q_table[state[0], state[1], action])
                state = next_state
                total_reward += reward
                if reward == 1:
                    house[next_state] = 0  # 
                    episode_info.append(f"Episode {episode}, Step {steps}: Robot found dirt at position {state}!")
            steps += 1

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_log.append(f"Episode {episode} completed with total reward: {total_reward}")
        episode_log.extend(episode_info)

    fig, ax = plt.subplots()
    ax.plot(rewards_per_episode)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward per Episode during Training')
    plt.savefig("training_rewards.png")
    plt.close()
    
    return "training_rewards.png", "\n".join(episode_log)

def simulate_robot(simulation_grid, iterations):
    global house, vacuum_pos
    house = np.array(simulation_grid, dtype=int)
    vacuum_pos = tuple(np.argwhere(house == 3)[0]) if 3 in house else None
    filenames = []

    state = vacuum_pos

    dirt_cleaned = 0
    start_time = time.time()

    for iteration in range(iterations):
        action = choose_action(state, epsilon=0)  # 
        next_state = get_next_state(state, action)
        if is_valid_state(next_state):
            state = next_state
        # 
        if house[state[0], state[1]] == 1:
            house[state[0], state[1]] = 0  # 
            dirt_cleaned += 1
        draw_grid(house, state, iteration)
        filenames.append(f'iteration_{iteration}.png')
        time.sleep(0.1)

    end_time = time.time()
    total_time = end_time - start_time

    # 
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('simulation.gif', images, duration=0.5)

    # 
    for filename in filenames:
        os.remove(filename)

    metrics = f'Total dirt cleaned: {dirt_cleaned}\n'
    metrics += f'Total simulation time: {total_time:.2f} seconds\n'
    metrics += f'Average dirt cleaned per iteration: {dirt_cleaned / iterations:.2f}'

    return 'simulation.gif', metrics

with gr.Blocks() as gui:
    gr.Markdown("# Vacuum Cleaner Robot Simulation\n**Created by Prof. Ramon Mayor Martins, Ph.D. [version 2.0 07/07/2024]**\n\n0-Read the instruction, 1-Set up the environment, 2-train the robot vacuum cleaner and 3-simulate.")
    
    with gr.Accordion("📋 Instructions", open=False):
        gr.Markdown("""
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
        """)

    with gr.Accordion("🏠⚙️ Environment Configuration", open=False):
        with gr.Row():
            with gr.Column():
                env_grid = gr.DataFrame(value=house.tolist(), headers=[str(i) for i in range(grid_size[1])], type="array", label="Edit the grid: 0 = Empty, 1 = Dirt, 2 = Wall, 3 = Vacuum")
                generate_button = gr.Button("Generate Environment")
            with gr.Column():
                env_img = gr.Image(interactive=False)
        generate_button.click(fn=update_grid, inputs=env_grid, outputs=env_img)

    with gr.Accordion("🤖🔧 Vacuum Robot Training", open=False):
        with gr.Row():
            episodes = gr.Number(label="Episodes", value=2000)
            alpha = gr.Number(label="Alpha (Learning Rate)", value=0.2)
            gamma = gr.Number(label="Gamma (Discount Factor)", value=0.95)
            epsilon = gr.Number(label="Epsilon (Exploration Rate)", value=1.0)
            epsilon_decay = gr.Number(label="Epsilon Decay", value=0.999)
            epsilon_min = gr.Number(label="Epsilon Min", value=0.1)
            move_penalty = gr.Number(label="Move Penalty", value=-0.1)
            train_button = gr.Button("Train Robot")
        with gr.Row():
            training_img = gr.Image(interactive=False)
            episode_log_output = gr.Textbox(label="Episode Log", lines=20, interactive=False)
        train_button.click(fn=train_robot, inputs=[episodes, alpha, gamma, epsilon, epsilon_decay, epsilon_min, move_penalty], outputs=[training_img, episode_log_output])

    with gr.Accordion("🤖📊 Robot Simulation", open=False):
        with gr.Row():
            new_simulation_grid = gr.DataFrame(value=house.tolist(), headers=[str(i) for i in range(grid_size[1])], type="array", label="New Simulation Grid: 0 = Empty, 1 = Dirt, 2 = Wall, 3 = Vacuum")
            iterations = gr.Number(label="Iterations", value=50)
            simulate_button = gr.Button("Simulate Robot")
        with gr.Row():
            simulation_img = gr.Image(interactive=False)
            metrics_output = gr.Textbox(label="Simulation Metrics", lines=10, interactive=False)
        simulate_button.click(fn=simulate_robot, inputs=[new_simulation_grid, iterations], outputs=[simulation_img, metrics_output])

gui.launch(debug=True)
