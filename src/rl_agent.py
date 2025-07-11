import random
from typing import Dict, Tuple, List

class RLAgent:
    """Simple Q-learning agent for routing decisions."""

    def __init__(self, learning_rate: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[Tuple[int, int], Dict[int, float]] = {}

    def choose_action(self, state: Tuple[int, int], actions: List[int]) -> int:
        self.q_table.setdefault(state, {a: 0.0 for a in actions})
        if random.random() < self.epsilon:
            return random.choice(actions)
        # choose action with max Q value
        return max(actions, key=lambda a: self.q_table[state].get(a, 0.0))

    def update(self, state: Tuple[int, int], action: int, reward: float, next_state: Tuple[int, int], next_actions: List[int]):
        self.q_table.setdefault(state, {})
        self.q_table[state].setdefault(action, 0.0)
        self.q_table.setdefault(next_state, {a: 0.0 for a in next_actions})
        max_next = max(self.q_table[next_state].values(), default=0.0)
        old_value = self.q_table[state][action]
        self.q_table[state][action] = old_value + self.lr * (reward + self.gamma * max_next - old_value)
