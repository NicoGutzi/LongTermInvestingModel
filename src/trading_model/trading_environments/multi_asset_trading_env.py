import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MultiAssetTradingEnv(gym.Env):
    """
    Custom Trading Environment for trading multiple indices (assets).

    Features:
    - Trade multiple indices simultaneously.
    - For each asset, the agent can decide to buy, sell, or hold.
    - Incorporates transaction costs, which can be set as either a percentage or a fixed cost.
    - Supports fractional shares.
    - Uses technical indicators and macroeconomic data for decision making.
    - Logs performance data for analysis.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data_dict, initial_cash=10000, 
                 buy_cost_pct: float = 0.001, sell_cost_pct: float = 0.001,
                 buy_cost_fixed: float = None, sell_cost_fixed: float = None):
        """
        Parameters:
        - data_dict: Dictionary of DataFrames, where each key is the asset name and the DataFrame 
                     contains columns such as 'open', 'close', etc. All DataFrames must have the same date index.
        - initial_cash: Starting cash balance.
        - buy_cost_pct: Transaction cost percentage when buying (used if fixed cost is None).
        - sell_cost_pct: Transaction cost percentage when selling (used if fixed cost is None).
        - buy_cost_fixed: Fixed cost per buy transaction (if provided, percentage is ignored).
        - sell_cost_fixed: Fixed cost per sell transaction (if provided, percentage is ignored).
        """
        super(MultiAssetTradingEnv, self).__init__()
        
        self.data_dict = data_dict
        self.assets = list(data_dict.keys())
        self.num_assets = len(self.assets)
        self.initial_cash = initial_cash
        
        # Transaction cost options: only one should be used.
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.buy_cost_fixed = buy_cost_fixed
        self.sell_cost_fixed = sell_cost_fixed
        
        self.dates = list(data_dict.values())[0].index
        self.num_steps = len(self.dates)
        self.current_step = 0
        
        # For each asset, state features: [price, moving_average, interest_rate, inflation_rate, unemployment_rate, ...]
        self.features_per_asset = 22  # TODO: Make dynamic if needed
        self.state_size = 1 + (self.num_assets * 2) + (self.num_assets * self.features_per_asset)
        
        # Action space: one continuous value per asset in [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        
        self.reset()
    
    def _get_asset_features(self, asset, date):
        """
        Retrieve features for a given asset at a given date.
        Dynamically extracts features for all prefixes found in the row.
        A prefix is defined as the part of a column name before an underscore ('_').
        If a column name has no underscore, it is used as its own prefix.
        """
        row = self._get_valid_row(asset, date)
        
        # Collect unique prefixes from all column names
        prefixes = set()
        for col in row.index:
            if "_" in col:
                prefixes.add(col.split("_")[0])
            else:
                prefixes.add(col)
        
        # For reproducibility, sort the prefixes and then, for each prefix,
        # sort the matching columns and add their values to the features list.
        features = []
        for prefix in sorted(prefixes):
            matching_cols = sorted([col for col in row.index if col.startswith(prefix)])
            for col in matching_cols:
                features.append(row[col])
        
        return np.array(features, dtype=np.float32)
    
    def _get_observation(self):
        """
        Build the state vector.
        State includes:
        - Cash (scalar)
        - Holdings for each asset (num_assets)
        - For each asset, its features (num_assets * features_per_asset)
        """
        date = self.dates[self.current_step]
        obs = [self.cash]
        
        # Append current holdings for each asset
        for asset in self.assets:
            obs.append(self.holdings[asset])
        
        # Append features for each asset
        for asset in self.assets:
            asset_features = self._get_asset_features(asset, date)
            obs.extend(asset_features.tolist())
        
        return np.array(obs, dtype=np.float32)
    
    def reset(self):
        """
        Resets the environment to the initial state.
        """
        self.cash = self.initial_cash
        self.holdings = {asset: 0 for asset in self.assets}
        self.current_step = 0
        self.total_asset = self.cash  # total portfolio value: cash + sum(holdings * price)
        self.asset_memory = [self.total_asset]
        self.action_history = []  # to log actions at each step
        self.trade_log = []       # to log individual trades (buy/sell)
        return self._get_observation()
    
    def step(self, actions):
        """
        Execute one time step within the environment.
        
        Parameters:
        - actions: A numpy array of shape (num_assets,) with values in [-1, 1].
          For each asset:
            - Positive value indicates buying, scaled by available cash.
            - Negative value indicates selling, scaled by holdings.
        
        Returns:
        - observation: Next state.
        - reward: Change in portfolio value.
        - done: Whether the episode is finished.
        - info: Additional info.
        """
        done = False
        date = self.dates[self.current_step]
        reward = 0
        
        # Record actions for visualization
        self.action_history.append(actions.copy())

        # Process actions for each asset
        for i, asset in enumerate(self.assets):
            row = self._get_valid_row(asset, date)
            current_price = row['open']
            action_value = actions[i]
            
            # If buying
            if action_value > 0:
                max_buyable = self.cash / current_price
                shares_to_buy = action_value * max_buyable
                shares_to_buy = max(shares_to_buy, 0)
                cost = shares_to_buy * current_price * (1 + self.buy_cost_pct)
                if shares_to_buy > 0:
                    # Use fixed cost if provided, else percentage cost.
                    if self.buy_cost_fixed is not None:
                        cost = shares_to_buy * current_price + self.buy_cost_fixed
                    else:
                        cost = shares_to_buy * current_price * (1 + self.buy_cost_pct)
                    if cost <= self.cash:
                        self.cash -= cost
                        self.holdings[asset] += shares_to_buy
                        self.trade_log.append({
                            "date": date,
                            "asset": asset,
                            "action": "buy",
                            "price": current_price,
                            "shares": shares_to_buy,
                            "cost": cost
                        })
            
            # If selling
            elif action_value < 0:
                shares_to_sell = abs(action_value) * self.holdings[asset]
                shares_to_sell = max(shares_to_sell, 0.0)
                if shares_to_sell > 0 and shares_to_sell <= self.holdings[asset]:
                    # Use fixed cost if provided, else percentage cost.
                    if self.sell_cost_fixed is not None:
                        revenue = shares_to_sell * current_price - self.sell_cost_fixed
                    else:
                        revenue = shares_to_sell * current_price * (1 - self.sell_cost_pct)
                        
                    self.cash += revenue
                    self.holdings[asset] -= shares_to_sell
                    self.trade_log.append({
                        "date": date,
                        "asset": asset,
                        "action": "sell",
                        "price": current_price,
                        "shares": shares_to_sell,
                        "revenue": revenue
                    })
        
        # Advance to the next step
        self.current_step += 1
        if self.current_step >= self.num_steps - 1:
            done = True
        
        # Calculate new portfolio value
        total_value = self.cash
        for asset in self.assets:
            row = self._get_valid_row(asset, self.dates[self.current_step])
            current_price = row['open']
            total_value += self.holdings[asset] * current_price
        previous_value = self.asset_memory[-1]
        reward = total_value - previous_value
        self.cash = self.cash  # update cash if needed
        self.total_asset = total_value
        self.asset_memory.append(total_value)
        
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, {}
    

    def _get_valid_row(self, asset, date):
        """
        Returns the data row for the given asset and date.
        If the date is missing, returns the row for the most recent available date.
        """
        df = self.data_dict[asset]
        if date in df.index:
            return df.loc[date]
        else:
            # Find the most recent date before the missing date
            valid_date = df.index[df.index < date].max()
            return df.loc[valid_date]
        
    def render(self, mode='human'):
        """
        Render the current state for inspection.
        """
        print(f"Step: {self.current_step}")
        print(f"Cash: {self.cash:.2f}")
        for asset in self.assets:
            print(f"{asset} Holdings: {self.holdings[asset]}")
        print(f"Total Asset Value: {self.total_asset:.2f}")
    
    def plot_trades(self):
        """
        Plot the trading decisions (actions) over time for each asset.
        """
        import matplotlib.pyplot as plt
        actions_array = np.array(self.action_history)  # shape: (timesteps, num_assets)
        time_steps = range(len(actions_array))
        
        plt.figure(figsize=(12, 8))
        for i, asset in enumerate(self.assets):
            plt.plot(time_steps, actions_array[:, i], label=f'{asset} Actions')
        plt.xlabel("Time Step")
        plt.ylabel("Action Value (-1: Sell, 0: Hold, 1: Buy)")
        plt.title("Trading Decisions Over Time")
        plt.legend()
        plt.show()


    def plot_performance(self):
        """
        Plot the portfolio value over time.
        """
        plt.plot(self.asset_memory)
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Value")
        plt.title("Portfolio Performance Over Time")
        plt.show()

    def get_trade_summary(self):
        """
        Return a summary table (DataFrame) of all trades.
        """
        return pd.DataFrame(self.trade_log)
    

    def plot_trades_on_price_chart(self, asset):
        """
        Overlay trades (buy/sell markers) on the asset's price chart.
        """
        # Get the asset's price data
        df = self.data_dict[asset].copy()
        df.index = pd.to_datetime(df.index)

        # Filter trades for this asset
        trades = [trade for trade in self.trade_log if trade["asset"] == asset]
        if not trades:
            print(f"No trades recorded for asset: {asset}")
            return

        # Separate trades into buys and sells
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
        plt.title(f"Trading Actions for {asset}")
        plt.legend()
        plt.show()