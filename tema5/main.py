import gymnasium as gym
import flappy_bird_gymnasium

import torch
import torch.nn as neural_network
import torch.optim as optimizers

import numpy as np
import cv2
import random
import matplotlib.pyplot as plot
import datetime

from collections import deque

# constants and hyperparameters

COMPUTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-4
DISCOUNT_FACTOR_GAMMA = 0.99 # how important future rewards are

MINI_BATCH_SIZE = 64
REPLAY_MEMORY_CAPACITY = 15000 # size of experience replay buffer

TARGET_NETWORK_UPDATE_FREQUENCY = 1000 

EPSILON_INITIAL = 1.0 # exploration
EPSILON_MINIMUM = 0.01 # exploitation but with a little chance of exploration
EPSILON_DECAY_RATE = 0.998

FRAME_HEIGHT_WIDTH = 84 # input image size for the neural network - standard for Atari games

# file names with timestamp to avoid overwriting
run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

MODEL_FILE_NAME = f"best_flappy_pixels_{run_timestamp}.pth"
PLOT_FILE_NAME = f"stats_{run_timestamp}.png"

def get_processed_game_frame():
    """
    Captures a frame from the game, converts it to grayscale,
    crops it, and resizes it for the neural network.
    """
    raw_rgb_frame = game_environment.render() # get raw pixels from the game

    grayscale_frame = cv2.cvtColor(
        raw_rgb_frame, cv2.COLOR_RGB2GRAY
    ) # convert to grayscale 

    frame_height, _ = grayscale_frame.shape
    cropped_frame = grayscale_frame[
        int(frame_height * 0.1) : int(frame_height * 0.85),
        :
    ] # crop to remove score and ground

    resized_frame = cv2.resize(
        cropped_frame, (FRAME_HEIGHT_WIDTH, FRAME_HEIGHT_WIDTH)
    ) # resize to 84x84 - uses the average of the nearest 4 pixels - bilinear interpolation

    normalized_frame = resized_frame / 255.0 # normalize pixel values between 0 and 1

    return normalized_frame.astype(np.float32)

class DuelingDeepQNetwork(neural_network.Module):
    def __init__(self, number_of_actions):
        super().__init__()

        # filter = feature

        self.convolutional_layers = neural_network.Sequential(
            neural_network.Conv2d(4, 32, kernel_size=8, stride=4), # 84x84 -> 20x20
            neural_network.ReLU(),

            neural_network.Conv2d(32, 64, kernel_size=4, stride=2), # 20x20 -> 9x9
            neural_network.ReLU(),

            neural_network.Conv2d(64, 64, kernel_size=3, stride=1), # 9x9 -> 7x7
            neural_network.ReLU(),

            neural_network.Flatten() # 64 * 7 * 7 = 3136
        )

        # Stream Value V(s)
        self.state_value_stream = neural_network.Sequential(
            neural_network.Linear(3136, 512),
            neural_network.ReLU(),
            neural_network.Linear(512, 1)
        )

        # Stream Advantage A(s,a)
        self.action_advantage_stream = neural_network.Sequential(
            neural_network.Linear(3136, 512),
            neural_network.ReLU(),
            neural_network.Linear(512, number_of_actions)
        )

    def forward(self, input_state_tensor):
        extracted_features = self.convolutional_layers(input_state_tensor)

        state_value = self.state_value_stream(extracted_features)
        action_advantage = self.action_advantage_stream(extracted_features)

        # Formula Dueling DQN
        q_values = state_value + (
            action_advantage
            - action_advantage.mean(dim=1, keepdim=True)
        )

        return q_values

game_environment = gym.make(
    "FlappyBird-v0",
    render_mode="rgb_array"
)

number_of_available_actions = game_environment.action_space.n # jump or do nothing

# We initialize the policy and target networks the same way

policy_network = DuelingDeepQNetwork(
    number_of_available_actions
).to(COMPUTE_DEVICE)

target_network = DuelingDeepQNetwork(
    number_of_available_actions
).to(COMPUTE_DEVICE)

# Initialize target network weights to match policy network
target_network.load_state_dict(
    policy_network.state_dict()
)

optimizer = optimizers.Adam(
    policy_network.parameters(),
    lr=LEARNING_RATE
)

# Experience replay buffer
replay_memory = deque(
    maxlen=REPLAY_MEMORY_CAPACITY
)

episode_rewards_history = []
episode_scores_history = []

best_score_achieved = 0
epsilon_greedy_value = EPSILON_INITIAL
total_training_steps = 0

print(f"Training started on: {COMPUTE_DEVICE}")

try:
    for episode_index in range(5000):

        game_environment.reset() # reset the environment at the start of each episode to restart game

        initial_frame = get_processed_game_frame()

        # A stack of 4 identical frames to represent the initial state
        current_state = np.stack(
            [initial_frame] * 4,
            axis=0
        )

        # Reward for the current episode initialized to zero
        cumulative_episode_reward = 0

        while True:

            if random.random() < epsilon_greedy_value:
                selected_action = game_environment.action_space.sample() # explore
            else:
                with torch.no_grad(): # exploit - infer action from policy network
                    state_tensor = torch.tensor(
                        current_state
                    ).unsqueeze(0).to(COMPUTE_DEVICE)
                    # unsqueeze to add batch dimension: 4x84x84 -> 1x4x84x84

                    selected_action = policy_network(
                        state_tensor
                    ).argmax().item()
                    # .item() to get the raw integer value in a Python scalar

            _, reward_received, terminated, truncated, info = (
                game_environment.step(selected_action)
            )

            done_episode = terminated or truncated

            next_frame = get_processed_game_frame()

            # Build the next state by appending the new frame and removing the oldest
            next_state = np.concatenate(
                (
                    current_state[1:],
                    np.expand_dims(next_frame, axis=0)
                ),
                axis=0
            )

            # Store the experience in the replay buffer
            replay_memory.append(
                (
                    current_state,
                    selected_action,
                    reward_received,
                    next_state,
                    done_episode
                )
            )

            current_state = next_state
            cumulative_episode_reward += reward_received # type: ignore
            total_training_steps += 1

            if len(replay_memory) >= MINI_BATCH_SIZE:

                mini_batch = random.sample(
                    replay_memory,
                    MINI_BATCH_SIZE
                )

                (
                    states_batch,
                    actions_batch,
                    rewards_batch,
                    next_states_batch,
                    done_flags_batch
                ) = zip(*mini_batch)

                states_tensor = torch.tensor(
                    np.array(states_batch)
                ).to(COMPUTE_DEVICE)

                actions_tensor = torch.tensor(
                    actions_batch
                ).to(COMPUTE_DEVICE)

                rewards_tensor = torch.tensor(
                    rewards_batch
                ).to(COMPUTE_DEVICE)

                next_states_tensor = torch.tensor(
                    np.array(next_states_batch)
                ).to(COMPUTE_DEVICE)

                done_flags_tensor = torch.tensor(
                    done_flags_batch
                ).to(COMPUTE_DEVICE)

                current_q_values = policy_network(
                    states_tensor
                ).gather(
                    1, actions_tensor.unsqueeze(1)
                ).squeeze()

                with torch.no_grad():
                    # choose the best action in the next state using the policy network
                    best_next_actions = policy_network(
                        next_states_tensor
                    ).argmax(1).unsqueeze(1)

                    # evaluate the Q value of that action using the target network
                    next_q_values = target_network(
                        next_states_tensor
                    ).gather(
                        1, best_next_actions
                    ).squeeze()

                    target_q_values = rewards_tensor + (
                        DISCOUNT_FACTOR_GAMMA
                        * next_q_values
                        * (1 - done_flags_tensor.float())
                    )

                loss_function = neural_network.SmoothL1Loss()
                loss = loss_function(
                    current_q_values,
                    target_q_values
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if total_training_steps % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                target_network.load_state_dict(
                    policy_network.state_dict()
                )

            if done_episode:
                break

        episode_score = info.get("score", 0)

        episode_rewards_history.append(
            cumulative_episode_reward
        )
        episode_scores_history.append(
            episode_score
        )

        if episode_score >= best_score_achieved:
            best_score_achieved = episode_score
            torch.save(
                policy_network.state_dict(),
                MODEL_FILE_NAME 
            )

        epsilon_greedy_value = max(
            EPSILON_MINIMUM,
            epsilon_greedy_value * EPSILON_DECAY_RATE
        )

        if episode_index % 10 == 0:
            print(
                f"Episode: {episode_index} | "
                f"Score: {episode_score} | "
                f"Best: {best_score_achieved} | "
                f"Epsilon: {epsilon_greedy_value:.3f}"
            )

except KeyboardInterrupt:
    print("\nAntrenare întreruptă manual.")

figure, (reward_axis, score_axis) = plot.subplots(2, 1, figsize=(10, 8))

reward_axis.plot(
    episode_rewards_history,
    color="blue",
    alpha=0.6
)
reward_axis.set_title("Evoluția recompensei")

score_axis.plot(
    episode_scores_history,
    color="green",
    alpha=0.6
)
score_axis.set_title("Țevi trecute per episod")

plot.tight_layout()
plot.savefig(PLOT_FILE_NAME) 
print(f"Statisticile au fost salvate în: {PLOT_FILE_NAME}")
plot.show()