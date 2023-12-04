import numpy as np
import monteCarlo as mc

class StudentMDPValueIteration(mc.StudentMDPMC):
    def __init__(self, discount_rate=0.99, epsilon=0.001):
        super().__init__()
        self.discount_rate = discount_rate
        self.epsilon = epsilon

    def value_iteration(self):
        num_iterations = 0
        while True:
            max_change = 0
            new_state_values = np.copy(self.state_action_values)  # Create a copy to store new values
            for i, state in enumerate(self.states):
                if state == 'terminal' or state == '8p':
                    continue

                # Calculate the expected values for all actions
                expected_values = [
                    self.calculate_expected_value(state, a) for a in self.actions
                ]

                # Update the new state values with the maximum expected value
                new_state_values[i, :] = expected_values

                # Calculate the maximum change for this state
                max_change = max(max_change, np.abs(self.state_action_values[i, 0] - new_state_values[i, 0]))

            # Update the state values after calculating changes for all states
            self.state_action_values = np.copy(new_state_values)

            num_iterations += 1

            # Print the state values after each iteration
            print(f"Iteration {num_iterations}:")
            self.print_results()

            # Check for convergence
            if max_change < self.epsilon:
                break

        print("\nNumber of iterations:", num_iterations)
        print("Final state values:")
        self.print_results()

        # Determine the optimal policy
        optimal_policy = []
        for i, state in enumerate(self.states):
            if state == 'terminal' or state == '8p':
                continue
            optimal_action = self.actions[np.argmax(self.state_action_values[i, :])]
            optimal_policy.append((state, optimal_action))

        print("\nFinal Optimal Policy:")
        print(optimal_policy)

    def calculate_expected_value(self, state, action):
        next_state, _ = self.transition(state, action)
        reward = self.get_reward(state, action, next_state)
        next_state_value = np.max(self.state_action_values[self.states.index(next_state), :])
        expected_value = reward + self.discount_rate * next_state_value
        return expected_value


student_mdp_value_iteration = StudentMDPValueIteration()
student_mdp_value_iteration.value_iteration()
