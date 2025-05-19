import random
from typing import Dict, List, Any, Tuple

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_model import SingleAssetTradingEnv
from src.utils.logger import logger

# --- Helper functions for a single asset ---
def get_random_window_for_asset(dates: List[pd.Timestamp], trading_years: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Select a random 5-year window from the available dates.
    """
    max_date = dates[-1]
    latest_possible_start = max_date - pd.DateOffset(years=trading_years)
    valid_start_dates = [d for d in dates if d <= latest_possible_start]
    random_start = random.choice(valid_start_dates)
    random_end = random_start + pd.DateOffset(years=trading_years)
    return random_start, random_end

def get_train_data_for_asset(data: pd.DataFrame, random_start: pd.Timestamp, random_end: pd.Timestamp) -> pd.DataFrame:
    """
    Get the subset of the asset data within the window.
    """
    return data[(data.index >= random_start) & (data.index < random_end)]

def create_train_env_asset(
    train_data: pd.DataFrame,
    initial_cash: float = 10000,
    buy_cost_pct: float = 0.01,
    sell_cost_pct: float = 0.01,
    buy_cost_fixed: float = None,
    sell_cost_fixed: float = None
) -> DummyVecEnv:
    """
    Create a new training environment for a single asset.
    
    Args:
        train_data: DataFrame for the training window.
        initial_cash: Starting cash.
        buy_cost_pct: Percentage cost for buying.
        sell_cost_pct: Percentage cost for selling.
        buy_cost_fixed: Fixed buy cost (if provided).
        sell_cost_fixed: Fixed sell cost (if provided).
    """
    if buy_cost_fixed is None:
        env = DummyVecEnv([lambda: SingleAssetTradingEnv(
            data=train_data, initial_cash=initial_cash, 
            buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct)])
    else:
        env = DummyVecEnv([lambda: SingleAssetTradingEnv(
            data=train_data, initial_cash=initial_cash, 
            buy_cost_fixed=buy_cost_fixed, sell_cost_fixed=sell_cost_fixed)])
    return env

def initialize_model_asset(
    data: pd.DataFrame,
    initial_cash: float = 10000,
    buy_cost_pct: float = 0.01,
    sell_cost_pct: float = 0.01,
    buy_cost_fixed: float = None,
    sell_cost_fixed: float = None
) -> PPO:
    """
    Initialize the PPO model with a dummy environment for a single asset.
    """
    if buy_cost_fixed is None:
        dummy_env = DummyVecEnv([lambda: SingleAssetTradingEnv(
            data=data, initial_cash=initial_cash, 
            buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct)])
    else:
        dummy_env = DummyVecEnv([lambda: SingleAssetTradingEnv(
            data=data, initial_cash=initial_cash, 
            buy_cost_fixed=buy_cost_fixed, sell_cost_fixed=sell_cost_fixed)])
    return PPO("MlpPolicy", dummy_env, device="cuda", verbose=1)

def train_on_window_asset(model: PPO, env: DummyVecEnv, total_timesteps: int) -> None:
    """
    Train the model on the given time window.
    """
    model.set_env(env)
    model.learn(total_timesteps=total_timesteps)

def log_iteration_result_asset(
    iteration: int,
    random_start: pd.Timestamp,
    random_end: pd.Timestamp,
    model_profit: float,
    bnh_profit: float
) -> Dict[str, Any]:
    """
    Log and return iteration results.
    """
    logger.info(f"Iteration {iteration}: Window {random_start.date()} to {random_end.date()} yields model profit: {model_profit:.2f} vs BnH profit: {bnh_profit:.2f}")
    return {
        "iteration": iteration,
        "start": random_start,
        "end": random_end,
        "model_profit": model_profit,
        "buy_and_hold_profit": bnh_profit
    }

def select_cost_params(
    fixed: bool,
    buy_cost_pct: float,
    sell_cost_pct: float,
    buy_cost_fixed: float,
    sell_cost_fixed: float
) -> Dict[str, float]:
    """
    Returns a dictionary of transaction cost parameters.
    """
    if fixed:
        return {"buy_cost_fixed": buy_cost_fixed, "sell_cost_fixed": sell_cost_fixed}
    else:
        return {"buy_cost_pct": buy_cost_pct, "sell_cost_pct": sell_cost_pct}
