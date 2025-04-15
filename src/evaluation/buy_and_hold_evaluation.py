from typing import Dict
import pandas as pd

def evaluate_buy_and_hold_multiple_assets(
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


def evaluate_buy_and_hold_single_asset(
    data: pd.DataFrame,
    initial_cash: float = 10000,
    buy_cost_pct: float = 0.01,
    sell_cost_pct: float = 0.01,
    buy_cost_fixed: float = None,
    sell_cost_fixed: float = None
) -> float:
    """
    Evaluates the buy-and-hold strategy for a single asset.
    
    The strategy invests the entire initial cash on the first day (using the 'open' price)
    and sells all shares on the last day. Transaction costs are applied either as a percentage 
    or as a fixed cost if provided.
    
    Args:
        data: DataFrame containing historical data for the asset, with a DateTime index.
        initial_cash: The initial cash available for investment.
        buy_cost_pct: Percentage transaction cost for buying (used if fixed cost is None).
        sell_cost_pct: Percentage transaction cost for selling (used if fixed cost is None).
        buy_cost_fixed: Fixed cost per buy transaction (if provided, percentage is ignored).
        sell_cost_fixed: Fixed cost per sell transaction (if provided, percentage is ignored).
    
    Returns:
        profit: The overall profit (or loss) from the buy-and-hold strategy.
    """
    # Sort the data by date
    data = data.sort_index()
    start_date = data.index[0]
    end_date = data.index[-1]
    
    # Get the opening prices on the first and last day.
    open_start = data.loc[start_date]['open']
    open_end = data.loc[end_date]['open']
    
    # Calculate the number of shares that can be bought.
    if buy_cost_fixed is not None:
        # Deduct the fixed cost first.
        shares = (initial_cash - buy_cost_fixed) / open_start if initial_cash > buy_cost_fixed else 0.0
    else:
        shares = initial_cash / (open_start * (1 + buy_cost_pct))
    
    # Compute the cost of purchase.
    if buy_cost_fixed is not None:
        cost = shares * open_start + buy_cost_fixed
    else:
        cost = shares * open_start * (1 + buy_cost_pct)
    
    # Leftover cash remains uninvested.
    leftover = initial_cash - cost
    
    # Compute revenue on selling.
    if sell_cost_fixed is not None:
        revenue = shares * open_end - sell_cost_fixed
    else:
        revenue = shares * open_end * (1 - sell_cost_pct)
    
    # Final portfolio value.
    final_value = revenue + leftover
    profit = final_value - initial_cash
    return profit