import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_model import MultiAssetTradingEnv

def walk_forward_validation(data_dict, train_years=5, val_years=1, total_timesteps=50000):
    """
    Performs walk-forward validation.
    
    Parameters:
    - data_dict: Dictionary of DataFrames with a DateTime index.
    - train_years: Number of years for the training period.
    - val_years: Number of years for the validation period.
    - total_timesteps: Timesteps for training the model in each fold.
    
    Returns:
    - results: Dictionary with tuple keys (train_start, train_end, val_end) and cumulative rewards as values.
    """
    # Assume all assets have the same date range.
    common_dates = list(data_dict[list(data_dict.keys())[0]].index)
    start_date = common_dates[0]
    end_date = common_dates[-1]
    
    current_train_start = start_date
    results = {}
    
    while True:
        train_end = current_train_start + pd.DateOffset(years=train_years)
        val_end = train_end + pd.DateOffset(years=val_years)
        
        # If validation period exceeds the available data, break the loop
        if val_end > end_date:
            break
        
        # Split the data into training and validation sets for each asset
        train_data = {
            asset: df[(df.index >= current_train_start) & (df.index < train_end)]
            for asset, df in data_dict.items()
        }
        val_data = {
            asset: df[(df.index >= train_end) & (df.index < val_end)]
            for asset, df in data_dict.items()
        }
        
        # Create training and validation environments
        train_env = DummyVecEnv([lambda: MultiAssetTradingEnv(train_data, initial_cash=10000, buy_cost_pct=0.01, sell_cost_pct=0.01)])
        val_env = DummyVecEnv([lambda: MultiAssetTradingEnv(val_data, initial_cash=10000, buy_cost_pct=0.01, sell_cost_pct=0.01)])
        
        # Train the PPO model on the training environment
        model = PPO("MlpPolicy", train_env, device="cuda", verbose=1)
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate the model on the validation environment
        total_reward = 0
        obs = val_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = val_env.step(action)
            total_reward += reward[0]
        
        # Log the results using the window time frame as key
        results[(current_train_start, train_end, val_end)] = total_reward
        print(f"Train period: {current_train_start.date()} to {train_end.date()}, " 
              f"Validation period: {train_end.date()} to {val_end.date()}, "
              f"Cumulative Reward: {total_reward:.2f}")
        
        # Move the training window forward (here, shifting by the training period length)
        current_train_start = train_end
        
    return results