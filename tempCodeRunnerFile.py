import random
import numpy as np

class StudentMDP:
    def __init__(self):
        self.states = ['R', 'T', 'D', 'U', '8p']
        self.actions = ['P', 'R', 'S', 'any']
        self.state_action_values = np.zeros((len(self.states), len(self.actions)))
        self.reward = []

    #  randomly selects an action from the list of actions
    def policy(self, state):
        return random.choice(self.actions)
    
    #follows the MDP's transition model to determine the next state and reward for a given state and action.
    def transition(self, state, action):
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(state, action, next_state)
        return next_state, reward

    def get_next_state(self, state, action):
        # Transition model
        if state == 'R':
            if action == 'P':
                return 'T'
            elif action == 'R':
                return 'R'
            elif action == 'S':
                return 'D'
            else:
                return 'terminal'
        elif state == 'T':
            if action == 'P':
                return 'D'
            elif action == 'R':
                return 'R'
            elif action == 'S':
                return 'U'
            else:
                return 'terminal'
        elif state == 'D':
            if action == 'P':
                return '8p'
            elif action == 'R':
                return 'T'
            elif action == 'S':
                return 'U'
            else:
                return 'terminal'
        elif state == 'U':
            if action == 'P':
                return '8p'
            elif action == 'R':
                return 'T'
            elif action == 'S':
                return 'U'
            else:
                return 'terminal'
        elif state == '8p':
            return '8p'

    def get_reward(self, state, action, next_state):
        # Rewards model
        if state == 'D' and action == 'S' and next_state == 'U':
            return 3
        elif state == '8p' and action == 'P' and next_state == '8p':
            return -2
        elif state == 'T' and action == 'R' and next_state == 'R':
            return 1
        else:
            return 0
    
    # runs an episode of the MDP and returns the sequence of experiences and the total reward for the episode
    def simulate_episode(self):
        state = random.choice(self.states)
        action = self.policy(state)
        episode = [(state, action)]
        while state != '8p':
            next_state, reward = self.transition(state, action)
            action = self.policy(next_state)
            episode.append((next_state, action, reward))
            state = next_state
        return episode, sum(reward for _, _, reward in episode if len(_) == 3)
    
    # implements first-visit Monte Carlo updates for updating the state-action values.
    def mc_update(self, episode):
        G = 0
        W = 1
        for state, action, _ in reversed(episode):
            G += W
            s_a_idx = self.states.index(state) * len(self.actions) + self.actions.index(action)
            self.state_action_values[s_a_idx] += 0.1 * (G - self.state_action_values[s_a_idx])
            W = W * 0.9
        
    # prints out the state-action values and the average reward per episode.
    def print_results(self):
        # Print header
        print("{:<10} {:<10} {:<15}".format("State", "Action", "Value"))
        for i, state in enumerate(self.states):
            for j, action in enumerate(self.actions):
                value = self.state_action_values[i * len(self.actions) + j]
                print("{:<10} {:<10} {:<15}".format(state, action, value))
        print("Average reward per episode: {}".format(sum(self.reward) / len(self.reward)))


student_mdp = StudentMDP()
for i in range(50):
    episode, episode_reward = student_mdp.simulate_episode()
    student_mdp.mc_update(episode)
    student_mdp.reward.append(episode_reward)
    student_mdp.print_results()