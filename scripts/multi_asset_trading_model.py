import sys
import os 
import random
from typing import Dict, List, Any, Tuple

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_model import preprocessing
from src.trading_model import MultiAssetTradingEnv
from src.utils.logger import logger
from src.evaluation.buy_and_hold_evaluation import evaluate_buy_and_hold_multiple_assets


# Function to evaluate the model on an environment (simulate one full episode)
def evaluate_model(env, model):
    total_reward = 0
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward[0] 
    return total_reward


def get_common_dates(data_dict: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    """Extract the common date index from one asset's DataFrame."""
    return list(data_dict[list(data_dict.keys())[0]].index)

def get_random_window(common_dates: List[pd.Timestamp], latest_possible_start: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return a random 5-year window given common dates and the latest start allowed."""
    valid_start_dates = [d for d in common_dates if d <= latest_possible_start]
    random_start = random.choice(valid_start_dates)
    random_end = random_start + pd.DateOffset(years=5)
    return random_start, random_end

def get_train_data(data_dict: Dict[str, pd.DataFrame], random_start: pd.Timestamp, random_end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """Return the training subset for each asset for the given window."""
    return {
        asset: df[(df.index >= random_start) & (df.index < random_end)]
        for asset, df in data_dict.items()
    }

def create_train_env(
    train_data: Dict[str, pd.DataFrame],
    initial_cash: float = 10000,
    buy_cost_pct: float = 0.01,
    sell_cost_pct: float = 0.01,
    buy_cost_fixed: float = None,
    sell_cost_fixed: float = None
) -> DummyVecEnv:
    """Create a new training environment for the given data."""
    if buy_cost_fixed is None:
        train_env = DummyVecEnv([lambda: MultiAssetTradingEnv(
            train_data, initial_cash=initial_cash, buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct)])
    else:
        train_env = DummyVecEnv([lambda: MultiAssetTradingEnv(
            train_data, initial_cash=initial_cash, buy_cost_fixed=buy_cost_fixed, sell_cost_fixed=sell_cost_fixed)])
    return train_env

def initialize_model(
    data_dict: Dict[str, pd.DataFrame],
    initial_cash: float = 10000,
    buy_cost_pct: float = 0.01,
    sell_cost_pct: float = 0.01,
    buy_cost_fixed: float = None,
    sell_cost_fixed: float = None
) -> PPO:
    """Initialize the PPO model with a dummy environment."""
    if buy_cost_fixed is None:
        dummy_env = DummyVecEnv([lambda: MultiAssetTradingEnv(
            data_dict=data_dict, initial_cash=initial_cash, buy_cost_pct=buy_cost_pct, sell_cost_pct=sell_cost_pct)])
    else:
        dummy_env = DummyVecEnv([lambda: MultiAssetTradingEnv(
            data_dict=data_dict, initial_cash=initial_cash, buy_cost_fixed=buy_cost_fixed, sell_cost_fixed=sell_cost_fixed)])
    return PPO("MlpPolicy", dummy_env, device="cuda", verbose=1)

def train_on_window(model: PPO, train_env: DummyVecEnv, total_timesteps: int) -> None:
    """Update the model by training on the given window."""
    model.set_env(train_env)
    model.learn(total_timesteps=total_timesteps)

def log_iteration_result(
    iteration: int,
    random_start: pd.Timestamp,
    random_end: pd.Timestamp,
    model_profit: float,
    bnh_profit: float
) -> Dict[str, Any]:
    """Log and return the results for a training window iteration."""
    logger.info(f"Window {random_start.date()} to {random_end.date()} yields model profit: {model_profit:.2f} vs buy & hold profit: {bnh_profit:.2f}")
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
    
    If fixed is True, returns fixed cost parameters; otherwise, returns percentage-based parameters.
    """
    if fixed:
        return {"buy_cost_fixed": buy_cost_fixed, "sell_cost_fixed": sell_cost_fixed}
    else:
        return {"buy_cost_pct": buy_cost_pct, "sell_cost_pct": sell_cost_pct}

def train_until_beat_buy_and_hold(
    data_dict: Dict[str, pd.DataFrame],
    required_success_rate: float = 0.6,
    total_attempts: int = 10,
    total_timesteps: int = 50000,
    initial_cash: float = 10000,
    fixed_transactions_cost: bool = False,
    buy_cost_pct: float = 0.01,
    sell_cost_pct: float = 0.01,
    buy_cost_fixed: float = 1,
    sell_cost_fixed: float = 1,
) -> List[Dict[str, Any]]:
    """
    Train the model until it beats a buy and hold strategy on random 5-year windows.
    First, when it beats the strategy for the first time, save a checkpoint.
    Then, require that it beats the buy & hold strategy in at least required_success_rate of windows 
    (e.g. 6/10) before finalizing the model.
    
    Args:
        data_dict: Dictionary of asset DataFrames.
        required_success_rate: Fraction of windows where the model must beat buy & hold.
        total_attempts: Number of windows to evaluate in each cycle.
        total_timesteps: Number of timesteps for each training iteration.
        initial_cash: Starting cash.
        fixed_transactions_cost: If True, use fixed cost model; otherwise, percentage-based.
        buy_cost_pct: Buying cost percentage (used if fixed_transactions_cost is False).
        sell_cost_pct: Selling cost percentage (used if fixed_transactions_cost is False).
        buy_cost_fixed: Fixed cost per buy (used if fixed_transactions_cost is True).
        sell_cost_fixed: Fixed cost per sell (used if fixed_transactions_cost is True).
    
    Returns:
        iteration_results: A list of dictionaries with details for each iteration.
    """
    # Choose the correct transaction cost parameters once
    cost_params = select_cost_params(
        fixed_transactions_cost, buy_cost_pct, sell_cost_pct, buy_cost_fixed, sell_cost_fixed
    )
    
    common_dates = get_common_dates(data_dict)
    max_date = common_dates[-1]
    latest_possible_start = max_date - pd.DateOffset(years=5)
    
    iteration = 0
    successes = 0
    attempts = 0
    iteration_results: List[Dict[str, Any]] = []
    success_rate_measure_start = False  # Flag to start measure success rate of model
    
    # Initialize the model using the selected cost parameters
    model = initialize_model(data_dict, initial_cash=initial_cash, **cost_params)
    
    while True:
        iteration += 1
        attempts += 1
        
        random_start, random_end = get_random_window(common_dates, latest_possible_start)
        logger.info(f"Iteration {iteration}: Training on window {random_start.date()} to {random_end.date()}")
        
        train_data = get_train_data(data_dict, random_start, random_end)
        new_train_env = create_train_env(train_data, initial_cash=initial_cash, **cost_params)
        train_on_window(model, new_train_env, total_timesteps)
        
        # Print the trade summary for debugging
        trade_summary = new_train_env.envs[0].get_trade_summary()
        logger.info(f"Trade summary for window {random_start.date()} to {random_end.date()}:\n{trade_summary}")
        
        # Evaluate the model and the buy & hold strategy on this window
        model_profit = evaluate_model(new_train_env, model)
        bnh_profit = evaluate_buy_and_hold_multiple_assets(train_data, initial_cash=initial_cash, **cost_params)
        
        iteration_results.append(log_iteration_result(iteration, random_start, random_end, model_profit, bnh_profit))
        
        if model_profit > bnh_profit:
            successes += 1

            # Start measuring success rate after 15 times beating bnh and save model
            if not success_rate_measure_start and successes >= 15:
                success_rate_measure_start = True
                successes = 1
                attempts = 1
                first_success_model_path = os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')),
                    "models", "multi_asset_trading_agent_first_success"
                )
                model.save(first_success_model_path)
                logger.info(f"Model saved after first success at window {random_start.date()} to {random_end.date()} at: {first_success_model_path}")
        
        logger.info(f"Current success rate: {successes} out of {attempts} attempts ({(successes/attempts)*100:.2f}%)")
        
        if attempts >= total_attempts and (successes / attempts) >= required_success_rate:
            final_model_path = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')),
                "models", "multi_asset_trading_agent"
            )
            model.save(final_model_path)
            logger.info(f"Final model saved with success rate {(successes/attempts)*100:.2f}% after {attempts} attempts at: {final_model_path}")
            break

    for result in iteration_results:
        print(f"Iteration {result['iteration']}: Window {result['start'].date()} to {result['end'].date()} - Model Profit: {result['model_profit']:.2f}, Buy & Hold Profit: {result['buy_and_hold_profit']:.2f}")
    
    return iteration_results

def train_model() -> None:
    """
    Run data ingestion and repeatedly train on random 5-year windows until the model beats the 
    buy & hold strategy in at least the required fraction of windows.
    """
    try:
        logger.info("Preprocess data.")
        preprocessed_data = preprocessing.preprocess_all_indices()
        logger.info("Finished preprocessing data successfully.")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        sys.exit(1)

    data_dict = preprocessed_data  # Each key is the asset name, and its value is a DataFrame

    # Here, required_success_rate=0.6 means the model must beat buy & hold in 6 out of 10 windows.
    train_until_beat_buy_and_hold(data_dict, required_success_rate=0.6, total_attempts=10, total_timesteps=50000, fixed_transactions_cost=True)

if __name__ == "__main__":
    train_model()