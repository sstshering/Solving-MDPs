import random
import numpy as np
import monteCarlo as mc

class StudentMDPQLearning(mc.StudentMDPMC):
    def __init__(self, discount_rate=0.99, learning_rate=0.2, epsilon=0.001):
        super().__init__()
        self.Q_values = np.zeros((len(self.states), len(self.actions)))
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
    def q_learning(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = random.choice(self.states)
            alpha = self.learning_rate

            while state != 'terminal' and state != '8p':
                action = self.policy(state)

                # Observe the reward and next state
                next_state, reward = self.transition(state, action)

                # Calculate the Q-value for the next state
                next_state_values = self.Q_values[self.states.index(next_state), :]
                next_state_max_value = np.max(next_state_values)

                # Q-learning update
                q_value = (1 - alpha) * self.Q_values[self.states.index(state), self.actions.index(action)] + \
                          alpha * (reward + self.discount_rate * next_state_max_value)

                # Print the previous and new Q-values, immediate reward, and Q-value for the next state
                print("Episode {}, State: {}, Action: {}".format(episode + 1, state, action))
                print(f"  Previous Q-value: {self.Q_values[self.states.index(state), self.actions.index(action)]}")
                print(f"  New Q-value: {q_value}")
                print(f"  Immediate Reward: {reward}")
                print(f"  Q-value for next state: {next_state_max_value}\n")

                # Update the Q-value
                self.Q_values[self.states.index(state), self.actions.index(action)] = q_value

                # Update the learning rate for the next iteration
                alpha *= 0.995

                state = next_state
                
                # Check for terminal state
                if state == 'terminal' or state == '8p':
                    break
                
    def print_results(self):
        print("Q-values:")
        print(self.Q_values)

        # Determine the optimal policy
        optimal_policy = []
        for i, state in enumerate(self.states):
            if state == 'terminal' or state == '8p':
                continue
            optimal_action = self.actions[np.argmax(self.Q_values[i, :])]
            optimal_policy.append((state, optimal_action))

            print("\nOptimal Policy:")
            print(optimal_policy)
            
  
student_mdp_q_learning = StudentMDPQLearning()
student_mdp_q_learning.q_learning(num_episodes=1000)
student_mdp_q_learning.print_results()