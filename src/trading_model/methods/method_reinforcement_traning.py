import os
from typing import Dict, List, Any, Tuple

import pandas as pd

from src.evaluation.buy_and_hold_evaluation import evaluate_buy_and_hold_single_asset
from src.trading_model import SingleAssetTradingEnv
from src.trading_model.methods.method_helper import select_cost_params, initialize_model_asset, get_random_window_for_asset, get_train_data_for_asset, create_train_env_asset, train_on_window_asset, log_iteration_result_asset

from src.utils.logger import logger

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
    dates = data.index.tolist()
    
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
