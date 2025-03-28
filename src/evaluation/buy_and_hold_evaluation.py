from typing import Dict
import pandas as pd

def evaluate_buy_and_hold(
    train_data: Dict[str, pd.DataFrame],
    initial_cash: float = 10000,
    buy_cost_pct: float = 0.01,
    sell_cost_pct: float = 0.01,
    buy_cost_fixed: float = None,
    sell_cost_fixed: float = None
) -> float:
    """
    Evaluates the buy and hold strategy on the given training data.
    For each asset, it invests an equal fraction of the cash on the first day 
    and sells on the last day, allowing fractional shares.
    
    Transaction costs are applied as either a percentage or a fixed cost 
    (if provided fixed cost is used exclusively).
    
    Returns:
        profit: Overall profit (which can be negative).
    """
    num_assets = len(train_data)
    cash_per_asset = initial_cash / num_assets
    final_value = 0.0

    for asset, df in train_data.items():
        df = df.sort_index()
        start_date = df.index[0]
        end_date = df.index[-1]

        # Get the open prices on the first and last day
        open_start = df.loc[start_date]['open']
        open_end = df.loc[end_date]['open']

        # Calculate fractional number of shares that can be bought
        if buy_cost_fixed is not None:
            # Solve: shares * open_start + fixed_cost = cash_per_asset
            shares = (cash_per_asset - buy_cost_fixed) / open_start if cash_per_asset > buy_cost_fixed else 0.0
        else:
            shares = cash_per_asset / (open_start * (1 + buy_cost_pct))
        
        # Compute cost of purchase
        if buy_cost_fixed is not None:
            cost = shares * open_start + buy_cost_fixed
        else:
            cost = shares * open_start * (1 + buy_cost_pct)
        
        # Compute revenue on selling
        if sell_cost_fixed is not None:
            revenue = shares * open_end - sell_cost_fixed
        else:
            revenue = shares * open_end * (1 - sell_cost_pct)
        
        # Leftover cash remains uninvested
        leftover = cash_per_asset - cost
        final_value += revenue + leftover

    profit = final_value - initial_cash
    return profit
