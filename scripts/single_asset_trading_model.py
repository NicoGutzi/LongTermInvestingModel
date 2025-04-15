import os
import random
import sys
from typing import Dict, List, Any, Tuple

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.trading_model import preprocessing
from src.trading_model import SingleAssetTradingEnv
from src.evaluation.buy_and_hold_evaluation import evaluate_buy_and_hold_single_asset
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

# --- Main training function for a single asset ---

def train_asset_until_beat_buy_and_hold(
    data: pd.DataFrame,
    asset_name:str,
    trading_years: int = 10,
    required_success_rate: float = 0.6,
    total_attempts: int = 10,
    total_timesteps: int = 50000,
    initial_cash: float = 10000,
    fixed_transactions_cost: bool = False,
    buy_cost_pct: float = 0.01,
    sell_cost_pct: float = 0.01,
    buy_cost_fixed: float = 1,
    sell_cost_fixed: float = 1,
    initial_success_threshold: int = 15
) -> List[Dict[str, Any]]:
    """
    Train a model for a single asset until it meets two criteria:
    1. It beats the buy & hold strategy in at least `initial_success_threshold` windows (initial phase).
    2. In an evaluation phase (over `total_attempts` windows), it beats buy & hold in at least
       `required_success_rate` of windows (e.g. 6/10).
       
    Args:
        data: DataFrame containing historical data for the asset.
        required_success_rate: Fraction of windows required in evaluation phase.
        total_attempts: Number of windows to evaluate in the evaluation phase.
        total_timesteps: Timesteps for training in each window.
        initial_cash: Starting cash.
        fixed_transactions_cost: Whether to use fixed cost model.
        buy_cost_pct: Percentage cost for buying (if fixed not used).
        sell_cost_pct: Percentage cost for selling (if fixed not used).
        buy_cost_fixed: Fixed buy cost (if used).
        sell_cost_fixed: Fixed sell cost (if used).
        initial_success_threshold: Number of successful windows (model profit > BnH profit) required in initial phase.
    
    Returns:
        List of iteration results.
    """
    dates = list(data)
    
    iteration = 0
    successes = 0
    attempts = 0
    iteration_results: List[Dict[str, Any]] = []
    evaluation_phase_started = False
    
    # Select cost parameters.
    cost_params = select_cost_params(fixed_transactions_cost, buy_cost_pct, sell_cost_pct, buy_cost_fixed, sell_cost_fixed)
    
    # Initialize model.
    model = initialize_model_asset(data, initial_cash=initial_cash, **cost_params)
    
    # --- Initial training Phase: achieve at least initial_success_threshold wins ---
    while not evaluation_phase_started:
        iteration += 1
        attempts += 1
        random_start, random_end = get_random_window_for_asset(dates, trading_years)
        logger.info(f"Iteration {iteration} (Initial Phase): Training on window {random_start.date()} to {random_end.date()}")
        
        train_data = get_train_data_for_asset(data, random_start, random_end)
        env = create_train_env_asset(train_data, initial_cash=initial_cash, **cost_params)
        train_on_window_asset(model, env, total_timesteps)
        
        learning_env: SingleAssetTradingEnv = env.envs[0]

        # Log trades
        trade_summary = learning_env.get_trade_summary()
        logger.info(f"Trade summary for window {random_start.date()} to {random_end.date()}:\n{trade_summary}")
        
        model_profit = learning_env.total_asset - learning_env.initial_cash
        bnh_profit = evaluate_buy_and_hold_single_asset(train_data, initial_cash=initial_cash, **cost_params)
        iteration_results.append(log_iteration_result_asset(iteration, random_start, random_end, model_profit, bnh_profit))
        
        if model_profit > bnh_profit:
            successes += 1
        
        logger.info(f"Initial Phase: {successes} wins out of {attempts} attempts")
        if successes >= initial_success_threshold:
            evaluation_phase_started = True
            successes = 0
            attempts = 0
    
    # --- Evaluation Phase: require at least required_success_rate wins over total_attempts ---
    while True:
        iteration += 1
        attempts += 1
        random_start, random_end = get_random_window_for_asset(dates)
        logger.info(f"Iteration {iteration} (Evaluation Phase): Training on window {random_start.date()} to {random_end.date()}")
        
        train_data = get_train_data_for_asset(data, random_start, random_end)
        env = create_train_env_asset(train_data, initial_cash=initial_cash, **cost_params)
        train_on_window_asset(model, env, total_timesteps)
        
        learning_env: SingleAssetTradingEnv = env.envs[0]

        # Log trades
        trade_summary = env.envs[0].get_trade_summary()
        logger.info(f"Trade summary for window {random_start.date()} to {random_end.date()}:\n{trade_summary}")
        
        model_profit = learning_env.total_asset - learning_env.initial_cash
        bnh_profit = evaluate_buy_and_hold_single_asset(train_data, initial_cash=initial_cash, **cost_params)
        iteration_results.append(log_iteration_result_asset(iteration, random_start, random_end, model_profit, bnh_profit))
        
        if model_profit > bnh_profit:
            successes += 1
        
        current_success_rate = successes / attempts
        logger.info(f"Evaluation Phase: {successes} wins out of {attempts} attempts ({current_success_rate*100:.2f}%)")
        
        if attempts >= total_attempts and current_success_rate >= required_success_rate:
            final_model_path = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')),
                "models", f"single_asset_trading_agent_{asset_name}")
            model.save(final_model_path)
            logger.info(f"Final model saved with success rate {current_success_rate*100:.2f}% after {attempts} attempts at: {final_model_path}")
            break
    
    for result in iteration_results:
        print(f"Iteration {result['iteration']}: Window {result['start'].date()} to {result['end'].date()} - Model Profit: {result['model_profit']:.2f}, BnH Profit: {result['buy_and_hold_profit']:.2f}")
    
    return iteration_results

def train_model() -> None:
    """
    Ingest data and train a model for each asset individually.
    Each asset's model is trained until it beats the buy & hold strategy 15 times,
    and then, in an evaluation phase, it must beat BnH in at least 6 out of 10 windows.
    """
    try:
        logger.info("Preprocess data.")
        preprocessed_data = preprocessing.preprocess_all_indices()
        logger.info("Finished preprocessing data successfully.")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        sys.exit(1)
    
    # preprocessed_data is a dict with asset names as keys.
    for asset, data in preprocessed_data.items():
        logger.info(f"Training model for asset: {asset}")
        train_results = train_asset_until_beat_buy_and_hold(data,
                                                             asset,
                                                             trading_years=10,
                                                             required_success_rate=0.6,
                                                             total_attempts=10,
                                                             total_timesteps=50000,
                                                             initial_cash=10000,
                                                             fixed_transactions_cost=True)
        logger.info(f"Training completed for asset: {asset}")
        # Optionally, save or analyze train_results for each asset.

if __name__ == "__main__":
    train_model()
