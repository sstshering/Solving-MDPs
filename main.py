import random
import numpy as np

class StudentMDP:
    def __init__(self):
        self.states = ['R', 'T', 'D', 'U', '8p','terminal']
        self.actions = ['P', 'R', 'S', 'any']
        self.state_action_values = np.zeros((len(self.states), len(self.actions)))
        self.reward = 0

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
                return 'U'
            elif action == 'R':
                return 'R'
            elif action == 'S':
                return 'D'
            else:
                return 'terminal'
        elif state == 'D':
            if action == 'P':
                return 'terminal'
            elif action == 'R':
                return 'T'
            elif action == 'S':
                return 'U'
            else:
                return 'terminal'
        elif state == 'U':
            if action == 'P':
                return 'terminal'
            elif action == 'R':
                return 'T'
            elif action == 'S':
                return 'U'
            else:
                return 'terminal'
        elif state == '8p':
            return 'terminal'
        else:
            return 'terminal'


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
        # print(state)
        action = self.policy(state)
        # print(action)
        episode = [(state, action, self.reward)]
        # print(episode)
        while state != 'terminal' and state != '8p':
            next_state, reward = self.transition(state, action)
            action = self.policy(next_state)
            reward = self.get_reward(state, action, next_state)
            print(reward)
            episode.append((next_state, action, reward))
            # print(episode)
            state = next_state
            
        print("Episode terminated")
        if len(episode) > 1:
            state, action, _ = episode[-2]
            next_state, action, reward = episode[-1]
            reward = self.get_reward(state, action, next_state)
            episode[-1] = (next_state, action, reward)
            
        total_reward = sum(r for _, _, r in episode)
        return episode, total_reward

    
    # implements first-visit Monte Carlo updates for updating the state-action values.
    def mc_update(self, episode):
        G = 0
        W = 1
        total_actions = len(self.actions)
        print("Ta", total_actions)
        for state, action, _ in reversed(episode):
            G += W
            s_a_idx = self.states.index(state) * total_actions + self.actions.index(action)
            print('s_a_idx', s_a_idx)
            print('state_action_values shape:', self.state_action_values.shape)
            self.state_action_values[s_a_idx] += 0.1 * (G - self.state_action_values[s_a_idx])
            W = W * 0.9


    # prints out the state-action values and the average reward per episode.
    def print_results(self):
        state_action_values_list = self.state_action_values.tolist()
        print("{:<10} {:<10} {:<15}".format("State", "Action", "Reward"))
        for i, state in enumerate(self.states):
            for j, action in enumerate(self.actions):
                values = state_action_values_list[i * len(self.actions) + j]
                total_value = sum(values)
                print("{:<10} {:<10} {:<15}".format(state, action, total_value))
        print("Average reward per episode: {:.2f}".format(sum(self.reward) / len(self.reward)))

student_mdp = StudentMDP()
for i in range(50):
    episode, episode_reward= student_mdp.simulate_episode()
    print("episode:\n", episode)
    # episode_reward = [sum(r for s, a, r in episode if len(episode) == 3)]
    print("episode reward:\n", episode_reward)
    student_mdp.mc_update(episode)
    student_mdp.reward += episode_reward
    student_mdp.print_results()