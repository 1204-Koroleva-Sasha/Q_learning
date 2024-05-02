# Alexandra Koroleva
# 04/23/2024
# Q learning (Off Policy control)
# Show the Q table of the last learning episode
# Plot the reward of all learning episodes


import numpy as np
import random
import matplotlib.pyplot as plt

# Constants for grid and goal
GRID_SIZE = 5
GOAL_POSITION = (4, 4)

# Q-learning parameters
ALPHA = 0.9
GAMMA = 0.9
EPSILON = 0.1

# Initialize Q-table
Q = np.random.rand(GRID_SIZE, GRID_SIZE, 4) * 0.02 - 0.01  # Small random values
ACTIONS = ['up', 'down', 'left', 'right']


def is_valid_position(position):
    """Check if the position is within grid boundaries."""
    x, y = position
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE


def take_action(state, action):
    """Move to new state based on action."""
    moves = {'up': (state[0], state[1] - 1), 'down': (state[0], state[1] + 1), 'left': (state[0] - 1, state[1]),
             'right': (state[0] + 1, state[1])}
    new_state = moves.get(action, state)
    return new_state if is_valid_position(new_state) else state


def choose_action(state, Q):
    """Choose action using epsilon-greedy policy."""
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)
    return ACTIONS[np.argmax(Q[state])]


def update_Q(state, action, reward, new_state, Q):
    """Update Q-value using the Q-learning update rule."""
    idx = ACTIONS.index(action)
    Q[state][idx] += ALPHA * (reward + GAMMA * np.max(Q[new_state]) - Q[state][idx])


def simulate_episode(Q):
    """Run one episode of Q-learning."""
    state = (0, 0)
    total_reward = 0
    details = []

    while state != GOAL_POSITION:
        action = choose_action(state, Q)
        new_state = take_action(state, action)
        reward = 100 if new_state == GOAL_POSITION else -1 if state == new_state else 0
        update_Q(state, action, reward, new_state, Q)
        state = new_state
        total_reward += reward
        details.append((state, action, reward))

    return total_reward, details


def print_q_values(Q):
    """Print Q-tables in a readable format."""
    print("Q-values (State: [Up, Down, Left, Right]):")
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            print(f"({x}, {y}): {Q[x, y].round(2).tolist()}")


def train_q_learning(num_episodes=12):
    episode_rewards = []
    initial_episode_details = None
    last_episode_details = None

    for episode in range(num_episodes):
        reward, details = simulate_episode(Q)
        episode_rewards.append(reward)
        if episode == 0:
            initial_episode_details = details
        if episode == num_episodes - 1:
            last_episode_details = details

    return episode_rewards, initial_episode_details, last_episode_details


def plot_rewards(episode_rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label='Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward of Learning Episodes')
    plt.legend()
    plt.show()


def main(num_episodes=12):
    # Train Q-learning and get results
    episode_rewards, initial_episode_details, last_episode_details = train_q_learning(num_episodes)

    print("Training complete :)")
    print_q_values(Q)

    plot_rewards(episode_rewards)


if __name__ == "__main__":
    main(12)