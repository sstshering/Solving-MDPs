import random

class State:
    # Define your states here
    EightPM = "8p"
    R = "R"
    T = "T"
    D = "D"
    U = "U"

class Action:
    # Define your actions here
    P = "P"
    R = "R"
    S = "S"
    Any = "Any"

class QLearning:
    def __init__(self, alpha, lambd):
        self.ALPHA = alpha
        self.LAMBDA = lambd
        self.QTable = {}
        self.initializeQTable()

    def initializeQTable(self):
        for state in State.__dict__.values():
            if isinstance(state, str):
                for action in Action.__dict__.values():
                    if isinstance(action, str):
                        self.QTable[(state, action)] = 0.0

    def chooseAction(self, state):
        possibleActions = self.getPossibleActions(state)
        randomIndex = random.randint(0, len(possibleActions) - 1)
        return possibleActions[randomIndex]

    def getPossibleActions(self, state):
        possibleActions = []
        for action in Action.__dict__.values():
            if isinstance(action, str) and self.isActionPossible(state, action):
                possibleActions.append(action)
        return possibleActions


    def isActionPossible(self, state, action):
        if state == State.EightPM:
            return action == Action.P or action == Action.R
        elif state == State.R:
            return action == Action.S or action == Action.Any
        elif state == State.T:
            return action == Action.R or action == Action.Any
        elif state == State.D:
            return action == Action.Any
        elif state == State.U:
            return action == Action.S or action == Action.Any
        else:
            return False

    def update_q_table(self, current_state, action, next_state, reward):
        old_value = self.QTable.get((current_state, action))
        max_value_next_state = max([self.QTable.get((next_state, a), 0.0) for a in Action.__dict__.values() if isinstance(a, str)])
        new_q_value = old_value + self.ALPHA * (reward + self.LAMBDA * max_value_next_state - old_value)
        self.QTable[(current_state, action)] = new_q_value

        
def terminal_state(state):
    return state == State.EightPM

def simulate_transition(current_state, action):
    next_state = State.T  # Replace with your logic
    reward = 0.5  # Replace with your logic
    return next_state, reward
    
# Test the Q-learning implementation
def test_q_learning():
    ql = QLearning(alpha=0.2, lambd=0.99)
    num_episodes = 50

    for episode in range(num_episodes):
        current_state = State.R
        total_reward = 0
        max_step = 100
        step = 0

        while not terminal_state(current_state) and step < max_step:
            action = ql.chooseAction(current_state)
            next_state, reward = simulate_transition(current_state, action)
            ql.update_q_table(current_state, action, next_state, reward)

            total_reward += reward
            current_state = next_state

            # Print relevant information after each Q-value update
            print(f"Episode {episode + 1}, Current State: {current_state}, Action: {action}, Next State: {next_state}, Reward: {reward}")
            for state_action, q_value in ql.QTable.items():
                print(f"{state_action}: {q_value}")
                
            step += 1

        # Print total reward for the episode
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # Print final Q-Values
    print("Final Q-Values:")
    for state_action, q_value in ql.QTable.items():
        print(f"{state_action}: {q_value}")

    # More testing code and evaluation if needed
    # ...

# Run the test
test_q_learning()
