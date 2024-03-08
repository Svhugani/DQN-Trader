from tensorflow.keras.models import Model
import numpy as np


class AgentMemoryBuffer:
    def __init__(self, state_dim: int, action_dim: int, length: int):
        self.states: np.ndarray = np.zeros([length, state_dim], dtype=np.float32)
        self.next_states: np.ndarray = np.zeros([length, state_dim], dtype=np.float32)
        self.actions: np.ndarray = np.zeros(length, dtype=np.uint16)
        self.rewards: np.ndarray = np.zeros(length, dtype=np.float32)
        self.dones: np.ndarray = np.zeros(length, dtype=bool)
        self.pointer: int = 0
        self.current_length: int = 0
        self.length: int = length

    def store_data(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.states[self.pointer] = state
        self.next_states[self.pointer] = next_state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.length
        self.current_length = min(self.current_length + 1, self.length)

    def sample_batch(self, batch_size: int = 32):
        batch_size = min(batch_size, self.length)
        idx = np.random.randint(0, self.length, size=batch_size)

        return {
            'states': self.states[idx],
            'next_states': self.next_states[idx],
            'actions': self.actions[idx],
            'rewards': self.rewards[idx],
            'dones': self.dones[idx]
        }


class DQNAgent:

    def __init__(self, state_dim: int, action_dim: int, memory_length: int = 500, model: Model = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = AgentMemoryBuffer(state_dim=state_dim, action_dim=action_dim, length=memory_length)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        if model is None:
            raise ValueError("A model is required for DQNAgent")
        self.model = model

    def update_agent_memory(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool):
        self.memory.store_data(state, action, reward, next_state, done)

    def take_action(self, state: np.array):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)

        action_values = self.model.predict(state.reshape((1, self.state_dim)))
        return np.argmax(action_values)

    def learn(self, batch_size: int = 32):
        if self.memory.current_length < batch_size:
            return

        batch = self.memory.sample_batch(batch_size)
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        target = rewards + (1 - dones) * self.gamma * np.amax(self.model.predict(next_states), axis=1)
        target_full = self.model.predict(states)
        target_full[np.arange(batch_size), actions] = target

        self.model.train_on_batch(states, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, model_name: str):
        self.model.load_weights(model_name)

    def save(self, model_name: str):
        self.model.save_weights(model_name)
