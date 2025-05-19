import sys

from src.trading_model import preprocessing
from src.utils.logger import logger
from src.trading_model.methods.method_unsupervised_learning import unsupervised_learning

from src.trading_model.methods import train_asset_until_beat_buy_and_hold

def train_model() -> None:
    """
    Ingest data and train a model for each asset individually.
    Each asset's model is trained until it beats the buy & hold strategy 15 times,
    and then, in an evaluation phase, it must beat BnH in at least 6 out of 10 windows.
    """
    # 1. load and preprocess data
    try:
        logger.info("Preprocess data.")
        preprocessed_data = preprocessing.preprocess_all_indices()
        logger.info("Finished preprocessing data successfully.")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        sys.exit(1)

    # 2. model training
    for financial_asset, financial_data in preprocessed_data.items():
        logger.info(f"Start unsupervised learning for asset: {financial_asset}")
        unsupervised_learning_result = unsupervised_learning(financial_data)
        logger.info(f"Unsupervised learning finished for asset: {financial_asset}")

        # reinforcement learning model
        logger.info(f"Training model for asset: {financial_asset}")
        train_results = train_asset_until_beat_buy_and_hold(unsupervised_learning_result,
                                                             financial_asset,
                                                             trading_years=10,
                                                             required_success_rate=0.6,
                                                             total_attempts=10,
                                                             total_timesteps=50000,
                                                             initial_cash=10000,
                                                             fixed_transactions_cost=True)
        logger.info(f"Training completed for asset: {financial_asset}")
    
    # preprocessed_data is a dict with asset names as keys.
    # for asset, data in unsupervised_learning_result.items():
    #     logger.info(f"Training model for asset: {asset}")
    #     train_results = train_asset_until_beat_buy_and_hold(data,
    #                                                          asset,
    #                                                          trading_years=10,
    #                                                          required_success_rate=0.6,
    #                                                          total_attempts=10,
    #                                                          total_timesteps=50000,
    #                                                          initial_cash=10000,
    #                                                          fixed_transactions_cost=True)
    #     logger.info(f"Training completed for asset: {asset}")
        # Optionally, save or analyze train_results for each asset.

if __name__ == "__main__":
    train_model()
