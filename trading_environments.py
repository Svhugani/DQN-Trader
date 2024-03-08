import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler


class BasicStockTradingEnv:
    """
    A Trading Environment

    State: Vector of size n * (1 + avg) + 1 where
        n - is the number of stocks
        avg - is the number of moving averages to calculate

    Action: Categorical variable with n^3 possibilities
        For each stock one of three cations can be performed:
        0 - HOLD
        1 - SELL
        2 - BUY
    """

    def __init__(self,
                 data: np.array,
                 use_returns: bool = False,
                 moving_averages: np.array = None,
                 init_balance: float = 20000):

        """
        Constructor for TradeEnv.

        Args:
            data (np.ndarray): The stock data represented by a NumPy array.
            use_returns (bool): Flag indicating whether to use returns format (default is False).
            moving_averages (np.ndarray): An array of moving averages to calculate (default is None).
            init_balance (float): Initial balance for trading (default is 20000).
        """

        self.returns_format: bool = use_returns
        self.moving_averages: np.array = moving_averages
        transformed_data = self._transform_data(data)
        self.stock_raw_data: np.ndarray = transformed_data[0]
        self.stock_data: np.ndarray = transformed_data[1]
        self.n_steps: int = self.stock_data.shape[0]
        self.n_stocks: int = data.shape[1]
        self.init_balance: float = init_balance
        self.action_space: list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stocks)))
        self.actions_dim: int = len(self.action_space)
        self.state_dim: int = self.stock_data.shape[1] + self.n_stocks + 1
        self.balance: float = None
        self.value: float = None
        self.current_step: int = None
        self.current_prices: np.array = None
        self.stock_wallet: np.array = None

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.init_balance
        self.value = self.balance
        self.stock_wallet = np.zeros(self.n_stocks)
        return self._observe()

    def step(self, action_id: int):
        if self.current_step == self.n_steps - 1:
            return

        prev_val = self.value
        self.current_step += 1
        self._update_current_prices()
        self._trade(action_id)
        self._update_current_value()

        reward = self.value - prev_val
        done = self.current_step == self.n_steps - 1
        info = {'current_value': self.value}

        return self._observe(), reward, done, info

    def get_scaler(self):
        states = []
        done = False
        self.reset()
        count = 0
        while not done:
            action = np.random.choice(self.actions_dim)
            state, reward, done, info = self.step(action)
            states.append(state)
            count += 1

        scaler = StandardScaler()
        scaler.fit(states)
        return scaler

    def _update_current_prices(self):
        self.current_prices = self.stock_raw_data[self.current_step]

    def _update_current_value(self):
        self.value = self.balance + np.dot(self.stock_wallet, self.current_prices)

    def _observe(self) -> np.array:
        state = np.empty(self.state_dim)
        state[:self.stock_data.shape[1]] = self.stock_data[self.current_step]
        state[self.stock_data.shape[1]:-1] = self.stock_wallet
        state[-1] = self.balance
        return state

    def _trade(self, action_id):
        action = self.action_space[action_id]
        stocks_to_sell_idx = [i for i, sub_action in enumerate(action) if sub_action == 1]
        stocks_to_buy_idx = [i for i, sub_action in enumerate(action) if sub_action == 2]

        for i in stocks_to_sell_idx:
            self.balance += self.current_prices[i] * self.stock_wallet[i]
            self.stock_wallet[i] = 0

        if len(stocks_to_buy_idx) == 0:
            return

        min_cash_to_buy = np.min(self.current_prices[stocks_to_buy_idx])
        while self.balance > min_cash_to_buy:
            for i in stocks_to_buy_idx:
                if self.balance > self.current_prices[i]:
                    self.stock_wallet[i] += 1
                    self.balance -= self.current_prices[i]

    def _transform_data(self, data: np.ndarray) -> (np.ndarray, np.ndarray):

        raw_data = data.copy()
        if self.returns_format:
            data = (data[1:] - data[: -1]) / data[: -1]

        if self.moving_averages is not None:
            n_row = data.shape[0] - self.moving_averages.max() + 1
            n_col = data.shape[1] * (self.moving_averages.shape[0] + 1)
        else:
            n_row = data.shape[0]
            n_col = data.shape[1]

        data_extend = np.empty((n_row, n_col))

        if self.moving_averages is not None:
            for j in range(data.shape[1]):
                col = j * (self.moving_averages.shape[0] + 1)
                data_extend[:, col] = data[-n_row:, j]

                for s, mov_size in enumerate(self.moving_averages):
                    data_extend[:, col + s + 1] = np.convolve(
                        data[:, j],
                        np.ones(mov_size) / mov_size,
                        mode='valid')[-n_row:]
        else:
            data_extend = data

        return raw_data[-n_row:], data_extend
