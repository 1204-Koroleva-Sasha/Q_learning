import unittest
from unittest.mock import patch
import random
import numpy as np
from io import StringIO
import sys
from q_learning import (is_valid_position, take_action, choose_action, update_Q,
                        simulate_episode, print_q_values,
                        train_q_learning, plot_rewards, main, GRID_SIZE)


# UNIT TESTS
class TestIsValidPosition(unittest.TestCase):

    def test_valid(self):
        self.assertTrue(is_valid_position((0, 0)))

    def test_out_of_bound_left(self):
        self.assertFalse(is_valid_position((-1, 0)))

    def test_out_of_bound_right(self):
        self.assertFalse(is_valid_position((1, GRID_SIZE)))

    def test_out_of_bound_up(self):
        self.assertFalse(is_valid_position((0, -1)))

    def test_out_of_bound_down(self):
        self.assertFalse(is_valid_position((0, GRID_SIZE)))


class TestTakeAction(unittest.TestCase):

    def test_up(self):
        self.assertEqual(take_action((2, 2), 'up'), (2, 1))

    def test_down(self):
        self.assertEqual(take_action((2, 2), 'down'), (2, 3))

    def test_left(self):
        self.assertEqual(take_action((2, 2), 'left'), (1, 2))

    def test_right(self):
        self.assertEqual(take_action((2, 2), 'right'), (3, 2))

    def test_boundary_left(self):
        self.assertEqual(take_action((0, 0), 'left'), (0, 0))

    def test_boundary_up(self):
        self.assertEqual(take_action((0, 0), 'up'), (0, 0))

    def test_boundary_right(self):
        self.assertEqual(take_action((GRID_SIZE, GRID_SIZE), 'right'), (GRID_SIZE, GRID_SIZE))

    def test_boundary_down(self):
        self.assertEqual(take_action((GRID_SIZE, GRID_SIZE), 'down'), (GRID_SIZE, GRID_SIZE))


class TestChooseAction(unittest.TestCase):
    def setUp(self):
        self.temp_q_table = {
            (0, 0): [0.1, 0.2, 0.3, 0.4],
            (1, 1): [0.5, 0.5, 0.5, 0.5],
            (2, 2): [-0.1, -0.2, -0.3, -0.4]

        }
        self.temp_actions = ['up', 'down', 'left', 'right']

    def test_choose_unequal_q_values(self):
        state = (0, 0)
        chosen_action = choose_action(state, self.temp_q_table)
        self.assertIn(chosen_action, self.temp_actions)

    def test_equal_q_values(self):
        state = (1, 1)
        chosen_action = choose_action(state, self.temp_q_table)
        self.assertIn(chosen_action, self.temp_actions)

    def test_negative_q_values(self):
        state = (2, 2)
        chosen_action = choose_action(state, self.temp_q_table)
        self.assertIn(chosen_action, self.temp_actions)


class TestUpdateQ(unittest.TestCase):
    def setUp(self):
        self.temp_q_table = {
            (0, 0): [0.1, 0.2, 0.3, 0.4],
            (0, 1): [0.0, 0.0, 0.0, 0.0]
        }

    def test_update_Q(self):
        initial_Q = self.temp_q_table.copy()

        initial_q_value = initial_Q[(0, 0)][0]

        update_Q((0, 0), 'up', 100, (0, 1), self.temp_q_table)

        self.assertNotEqual(self.temp_q_table[(0, 0)][0], initial_q_value)


# INTEGRATION TESTS
class TestQLearning(unittest.TestCase):
    def setUp(self):
        self.held_output = StringIO()
        sys.stdout = self.held_output

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def capture_print_output(self, func, *args, **kwargs):
        """Execute a function and capture its print output."""
        func(*args, **kwargs)
        return self.held_output.getvalue()

    def test_print_q_values(self):
        Q_test = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        Q_test[0, 0] = [0.1, 0.2, 0.3, 0.4]
        Q_test[4, 4] = [1, 1, 1, 1]

        actual_output = self.capture_print_output(print_q_values, Q_test)
        self.assertIn("(0, 0): [0.1, 0.2, 0.3, 0.4]", actual_output)
        self.assertIn("(4, 4): [1.0, 1.0, 1.0, 1.0]", actual_output)


class TestTrainQLearning(unittest.TestCase):

    @patch('q_learning.simulate_episode')
    def test_train_q_learning_returns_correct_values(self, mock_simulate_episode):
        mock_simulate_episode.side_effect = [
            (10, [('state1', 'action1', 10)]),  # Episode 1
            (-1, [('state2', 'action2', -1)]),  # Episode 2
            (20, [('state3', 'action3', 20)])  # Episode 3
        ]

        num_episodes = 3
        episode_rewards, initial_episode_details, last_episode_details = train_q_learning(num_episodes)

        # Check if the episode rewards are correct
        self.assertEqual(episode_rewards, [10, -1, 20])

        # Check if the first episode details are correctly captured
        self.assertEqual(initial_episode_details, [('state1', 'action1', 10)])

        # Check if the last episode details are correctly captured
        self.assertEqual(last_episode_details, [('state3', 'action3', 20)])

    @patch('q_learning.simulate_episode')
    def test_consistency_across_episodes(self, mock_simulate_episode):
        # Testing consistency in episode rewards computation
        consistent_reward = 5
        consistent_details = [('state', 'action', 5)]
        mock_simulate_episode.return_value = (consistent_reward, consistent_details)

        num_episodes = 5
        episode_rewards, _, _ = train_q_learning(num_episodes)

        self.assertTrue(all(reward == consistent_reward for reward in episode_rewards))


class TestMainFunction(unittest.TestCase):
    @patch('q_learning.plot_rewards')
    @patch('q_learning.print_q_values')
    @patch('q_learning.simulate_episode')
    @patch('builtins.print')
    def test_main(self, mock_print, mock_simulate_episode, mock_print_q_values, mock_plot_rewards):
        mock_simulate_episode.return_value = (50, ['details1', 'details2'])

        main(12)

        self.assertEqual(mock_simulate_episode.call_count, 12)

        mock_print.assert_any_call("Training complete :3")

        mock_print_q_values.assert_called_once()

        mock_plot_rewards.assert_called_once()


class TestPlotRewards(unittest.TestCase):
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.figure')
    def test_plot_rewards(self, mock_figure, mock_plot, mock_xlabel, mock_ylabel, mock_title, mock_legend, mock_show):
        episode_rewards = [10, 20, -5, 15, 25]
        plot_rewards(episode_rewards)

        mock_figure.assert_called_once_with(figsize=(10, 6))

        mock_plot.assert_called_once_with(episode_rewards, label='Total Reward')

        mock_xlabel.assert_called_once_with('Episode')
        mock_ylabel.assert_called_once_with('Total Reward')
        mock_title.assert_called_once_with('Reward of Learning Episodes')
        mock_legend.assert_called_once()

        mock_show.assert_called_once()


class TestPolicyEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp_q_table = np.random.rand(GRID_SIZE, GRID_SIZE, 4) * 0.02 - 0.01

    def test_policy_evaluation(self):
        num_episodes = 100
        total_rewards = []
        for _ in range(num_episodes):
            total_reward, _ = simulate_episode(self.temp_q_table)
            total_rewards.append(total_reward)

        self.assertGreaterEqual(total_rewards[-1], total_rewards[0])


class TestExplorationVsExploitation(unittest.TestCase):
    def setUp(self):
        self.temp_q_table = np.random.rand(GRID_SIZE, GRID_SIZE, 4) * 0.02 - 0.01

    def test_exploration_vs_exploitation(self):
        EPSILON = 0.1
        num_episodes = 100
        exploration_count = 0
        exploitation_count = 0

        for _ in range(num_episodes):
            state = (0, 0)
            for _ in range(10):
                if random.uniform(0, 1) < EPSILON:
                    exploration_count += 1
                else:
                    exploitation_count += 1
                action = choose_action(state, self.temp_q_table)
                state = take_action(state, action)

        total_actions = exploration_count + exploitation_count
        exploration_ratio = exploration_count / total_actions
        exploitation_ratio = exploitation_count / total_actions

        self.assertGreaterEqual(exploitation_ratio, 0.5)
        self.assertLessEqual(exploration_ratio, 0.9)


