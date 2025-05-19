import gym
from typing import Tuple
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SingleAssetTradingEnv(gym.Env):
    """
    Custom Trading Environment for trading a single index (asset).

    Features:
    - Trade one asset.
    - The agent decides when to buy, sell, or hold.
    - Incorporates transaction costs (either percentage or fixed).
    - Supports fractional shares.
    - Uses technical indicators and macroeconomic data for decision making.
    - Logs performance data for analysis.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data: pd.DataFrame, initial_cash: float = 10000, 
                 buy_cost_pct: float = 0.001, sell_cost_pct: float = 0.001,
                 buy_cost_fixed: float = None, sell_cost_fixed: float = None):
        """
        Parameters:
        - data: DataFrame containing historical data for the asset with a DateTime index.
        - initial_cash: Starting cash balance.
        - buy_cost_pct: Transaction cost percentage when buying (if fixed cost is None).
        - sell_cost_pct: Transaction cost percentage when selling (if fixed cost is None).
        - buy_cost_fixed: Fixed cost per buy transaction (if provided, percentage is ignored).
        - sell_cost_fixed: Fixed cost per sell transaction (if provided, percentage is ignored).
        """
        super(SingleAssetTradingEnv, self).__init__()
        
        self.data = data.sort_index()
        self.initial_cash = initial_cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.buy_cost_fixed = buy_cost_fixed
        self.sell_cost_fixed = sell_cost_fixed
        
        self.dates = self.data.index
        self.num_steps = len(self.dates)
        self.current_step = 0
        
        # For simplicity, we assume all columns (except the index) are features.
        # Adjust this as needed.
        self.features = list(self.data.columns)
        self.features_per_asset = len(self.features)
        
        # State: [cash, holdings, feature1, feature2, ... featureN]
        self.state_size = 1 + 1 + self.features_per_asset
        
        # Action space: scalar in [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        
        self.reset()
    
    def _get_features(self, date: pd.Timestamp) -> np.ndarray:
        """Extract features from the data for the given date."""
        row = self._get_valid_row(date)
        # Return all feature columns as a numpy array.
        return np.array(row[self.features].tolist(), dtype=np.float32)
    
    def _get_valid_row(self, date: pd.Timestamp) -> pd.Series:
        """
        Get the data row for a given date. If the date is missing,
        return the row for the most recent available date.
        """
        if date in self.data.index:
            return self.data.loc[date]
        else:
            valid_date = self.data.index[self.data.index < date].max()
            return self.data.loc[valid_date]
    
    def _get_observation(self) -> np.ndarray:
        """
        Build the observation vector from the current state.
        """
        date = self.dates[self.current_step]
        obs = [self.cash, self.holdings]
        obs.extend(self._get_features(date).tolist())
        return np.array(obs, dtype=np.float32)
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state.
        """
        self.cash = self.initial_cash
        self.holdings = 0.0  # Allow fractional holdings.
        self.current_step = 0
        self.total_asset = self.cash  # cash + holdings * price.
        self.asset_memory = [self.total_asset]
        self.action_history = []  # Logs actions.
        self.trade_log = []       # Logs trades.
        return self._get_observation()
    
    def _compute_reward(self, window_size: int = 5) -> float:
        """
        Compute a composite reward based on:
        1. Log return over a rolling window.
        2. A drawdown penalty over that window.
        3. A penalty if there is a significant market drop on the current step.
        
        Args:
            window_size: Number of steps over which to calculate the log return and drawdown.
        
        Returns:
            Composite reward value.
        """
        # 1) Compute log-return over the window
        # Use a window of past values if available, else fallback to daily change.
        if len(self.asset_memory) > window_size:
            previous_value = self.asset_memory[-window_size - 1]
        else:
            previous_value = self.asset_memory[-1]
        current_value = self.total_asset

        log_return = 0.0
        if previous_value > 0 and current_value > 0:
            log_return = np.log(current_value) - np.log(previous_value)
        
        # Compute drawdown penalty over the window.
        window = self.asset_memory[-window_size:]
        peak = max(window) if window else previous_value
        drawdown = (peak - current_value) / peak if peak > 0 else 0.0
        drawdown_penalty = -0.1 * drawdown  # adjust multiplier as needed
        
        # Combine the reward components.
        reward = log_return + drawdown_penalty
        return reward

    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one time step within the environment.
        
        Args:
            action: A numpy array of shape (1,) with a value in [-1, 1].
                    Positive values indicate buying; negative indicate selling.
        
        Returns:
            observation: The next state as a numpy array.
            reward: The change in portfolio value.
            done: Whether the episode has ended.
            info: Additional info (empty dict here).
        """
        done = False
        date = self.dates[self.current_step]
        reward = 0.0
        
        # Log the action.
        self.action_history.append(action.copy())
        
        row = self._get_valid_row(date)
        current_price = row['open']
        
        # Action interpretation: positive for buying, negative for selling.
        if action[0] > 0:  # Buying.
            max_buyable = self.cash / current_price  # fractional shares allowed.
            shares_to_buy = action[0] * max_buyable
            shares_to_buy = max(shares_to_buy, 0.0)
            if shares_to_buy > 0:
                if self.buy_cost_fixed is not None:
                    cost = shares_to_buy * current_price + self.buy_cost_fixed
                else:
                    cost = shares_to_buy * current_price * (1 + self.buy_cost_pct)
                if cost <= self.cash:
                    self.cash -= cost
                    self.holdings += shares_to_buy
                    self.trade_log.append({
                        "date": date,
                        "action": "buy",
                        "price": current_price,
                        "shares": shares_to_buy,
                        "cost": cost
                    })
                    
        elif action[0] < 0:  # Selling.
            shares_to_sell = abs(action[0]) * self.holdings
            shares_to_sell = max(shares_to_sell, 0.0)
            if shares_to_sell > 0 and shares_to_sell <= self.holdings:
                if self.sell_cost_fixed is not None:
                    revenue = shares_to_sell * current_price - self.sell_cost_fixed
                else:
                    revenue = shares_to_sell * current_price * (1 - self.sell_cost_pct)
                self.cash += revenue
                self.holdings -= shares_to_sell
                self.trade_log.append({
                    "date": date,
                    "action": "sell",
                    "price": current_price,
                    "shares": shares_to_sell,
                    "revenue": revenue
                })
        
        self.current_step += 1
        if self.current_step >= self.num_steps - 1:
            done = True
        
        # Calculate new portfolio value.
        next_price = self._get_valid_row(self.dates[self.current_step])['open']
        total_value = self.cash + self.holdings * next_price
        self.total_asset = total_value
        self.asset_memory.append(total_value)
        
        # Use the new composite reward function.
        reward = self._compute_reward(window_size=15)
        
        # Build observation
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), done, {}
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Cash: {self.cash:.2f}")
        print(f"Holdings: {self.holdings:.4f}")
        print(f"Total Asset Value: {self.total_asset:.2f}")
    
    def plot_performance(self):
        plt.plot(self.asset_memory)
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Value")
        plt.title("Portfolio Performance Over Time")
        plt.show()
    
    def get_trade_summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)
    
    def plot_trades(self):
        actions_array = np.array(self.action_history)
        time_steps = range(len(actions_array))
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, actions_array[:, 0], label="Actions")
        plt.xlabel("Time Step")
        plt.ylabel("Action Value (-1: Sell, 0: Hold, 1: Buy)")
        plt.title("Trading Actions Over Time")
        plt.legend()
        plt.show()
    
    def plot_trades_on_price_chart(self):
        df = self.data.copy()
        df.index = pd.to_datetime(df.index)
        trades = self.trade_log
        if not trades:
            print("No trades recorded.")
            return
        buy_dates = [trade["date"] for trade in trades if trade["action"] == "buy"]
        buy_prices = [trade["price"] for trade in trades if trade["action"] == "buy"]
        sell_dates = [trade["date"] for trade in trades if trade["action"] == "sell"]
        sell_prices = [trade["price"] for trade in trades if trade["action"] == "sell"]
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["open"], label="Open Price", alpha=0.7)
        if buy_dates:
            plt.scatter(buy_dates, buy_prices, color="green", marker="^", s=100, label="Buy")
        if sell_dates:
            plt.scatter(sell_dates, sell_prices, color="red", marker="v", s=100, label="Sell")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Trading Actions")
        plt.legend()
        plt.show()