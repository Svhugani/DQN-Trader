import argparse
import sys
import tensorflow as tf
import numpy as np
import utils
from trading_environments import BasicStockTradingEnv
from agents import DQNAgent
from learning_models import standard_model
from datetime import datetime
import pickle
from matplotlib import pyplot as plt

MODELS_FOLDER = 'ai_trading_models'
RESULTS_FOLDER = 'ai_trading_rewards'


def print_results_agent():
    results = np.load(f'{RESULTS_FOLDER}/wallet_values.npy')
    plt.plot(results)
    plt.grid()
    plt.title("Trading results")
    plt.show()


def run_trading_agent(
        data_path: str,
        train_mode: bool = False,
        train_amount: float = 0.6,
        n_episodes: int = 2000,
        batch_size: int = 32,
        initial_balance: float = 1000,
        moving_averages: np.array = None):

    utils.try_make_dir(MODELS_FOLDER)
    utils.try_make_dir(RESULTS_FOLDER)

    data = utils.get_raw_data(data_path)

    if data is None:
        raise FileNotFoundError("Couldn't find a correct file!")

    n_steps, n_stocks = data.shape
    n_train_steps = int(train_amount * n_steps)

    train_data = data[:n_train_steps]
    test_data = data[n_train_steps:]

    env = BasicStockTradingEnv(
        data=train_data if train_mode else test_data,
        use_returns=True,
        moving_averages=moving_averages,
        init_balance=initial_balance)

    scaler = env.get_scaler()
    model = standard_model(input_dim=env.state_dim, output_dim=env.actions_dim, hidden_layers=2, hidden_dim=32)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.actions_dim, memory_length=500, model=model)

    wallet_values = []

    if not train_mode:
        with open(f'{MODELS_FOLDER}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        agent.epsilon = 0.025
        agent.load(f'{MODELS_FOLDER}/dqn.h5')

    print("Initialize episodes")
    for episode in range(n_episodes):
        start_time = datetime.now()
        state = scaler.transform([env.reset()])
        done = False

        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = scaler.transform(next_state.reshape(1, -1))
            if train_mode:
                agent.update_agent_memory(state, action, reward, next_state, done)
                agent.learn(batch_size)
            state = next_state

        value = info['current_value']

        time_delta = datetime.now() - start_time
        print(f"Episode: {episode + 1} / {n_episodes}. Wallet value: {value:.2f}, time delta: {time_delta}")
        wallet_values.append(value)

    if train_mode:
        agent.save(f'{MODELS_FOLDER}/dqn.h5')

        with open(f'{MODELS_FOLDER}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    np.save(f'{RESULTS_FOLDER}/wallet_values.npy', wallet_values)


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    tf.config.run_functions_eagerly(False)
    run_trading_agent(
        data_path='aapl_msi_sbux.csv',
        n_episodes=2000,
        train_mode=True,
        batch_size=64,
        initial_balance=5000,
        moving_averages=np.array([5, 15, 30]))


    print_results_agent()
