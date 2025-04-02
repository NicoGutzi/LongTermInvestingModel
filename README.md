# LongTermInvestingModel

This repository is dedicated to training a long-term investing model that leverages historical price data and economic indicators. The primary focus is on modeling indices rather than individual stocks. By integrating data from multiple sources and applying both unsupervised and reinforcement learning techniques, the model aims to identify market trends and make informed trading decisions that can outperform traditional buy-and-hold strategies.

## Table of Contents

- [Overview](#overview)
- [Data Sources](#data-sources)
- [Workflow](#workflow)
- [Data Preprocessing](#data-preprocessing)
- [Modeling Approach](#modeling-approach)
- [Future Improvements](#future-improvements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project trains an investing model using a combination of:
- **Historical price data**: Fetched from Yahoo Finance.
- **Economic indicators**: Includes interest rates, inflation rates, and unemployment rates from multiple regions (USA, EU, Japan, China, Brazil, India) obtained from the Federal Reserve Economic Data (FRED).

The system is designed to automatically handle multiple indices and indicators, enabling robust analysis across diverse market conditions.

## Data Sources

- **Price Data**: Sourced from Yahoo Finance, focusing on major indices.
- **Economic Indicators**: Retrieved from FRED, covering:
  - Interest Rates
  - Inflation Rates
  - Unemployment Rates
- **Flags**: Special columns (e.g., `inflation_rate_CN_flag`) indicate data integrity, helping the model account for missing or incomplete indicator data.

## Workflow

The project follows these key steps:

1. **Data Request**: Fetch historical price data and economic indicators.
2. **Data Storage**: Save the retrieved data into a local PostgreSQL database (managed via pgAdmin).
3. **Data Loading**: Extract data from the database using SQL queries.
4. **Data Preprocessing**:
   - **Alignment**: Convert monthly indicator data to daily frequency by forward-filling the latest available value.
   - **Missing Values**: Use flag indicators to mark and manage incomplete data.
   - **Normalization**: Address mixed units (e.g., raw prices vs. percentage rates) to improve model performance.
5. **Model Training**: 
   - Train an unsupervised learning model to recognize market trends.
   - Integrate these trends into a reinforcement learning model to generate trading signals.

## Data Preprocessing

- **Timeframe Alignment**: Since price data is daily and some economic indicators are monthly, the monthly data is forward-filled to match the daily time series.
- **Handling Missing Data**: Flag indicators are added to denote missing values, ensuring the model is aware of data integrity issues.
- **Normalization**: Due to mixed units (prices vs. percentage rates), data normalization is essential for effective learning.

## Modeling Approach

### Unsupervised Learning
- **Objective**: Recognize and cluster market trends by extracting latent representations from historical data.
- **Techniques**: 
   TODO

### Reinforcement Learning
- **Objective**: Utilize the trends and latent representations from the unsupervised model as inputs for decision-making.
- **Approach**: Train an RL agent (e.g., using PPO) on a custom `gym.Env` to optimize trading strategies over long-term horizons.
- **Reward Structure**: Designed to encourage long-term profitability while managing risks and transaction costs.

## Future Improvements

- **Market Trend Identification**: Enhance unsupervised learning methods to better detect market regimes and integrate these insights into the RL model.
- **Automated Pipelines**: Develop robust data and model training pipelines for seamless automation.
- **Data Normalization**: Further refine normalization strategies to account for mixed units and improve learning efficiency.
- **Performance Benchmarking**: Compare model performance against a traditional buy-and-hold strategy.
- **Advanced Architectures**: Experiment with recurrent or transformer-based policy networks for improved long-term memory and sequence processing.

## Installation
This project uses **Docker Compose** to build and run all required services. Follow these steps to set up your environment:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/LongTermInvestingModel.git
   cd LongTermInvestingModel

2. **Setup the .env file**
3. **Build the docker file**:
    ```bash
   docker-compose up --build




## Usage

- **Data Acquisition**: Run the data fetching scripts to download historical price data and economic indicators.
- **Data Storage**: Data is automatically saved into your local PostgreSQL database.
- **Preprocessing**: Execute the preprocessing scripts to align, normalize, and flag the data.
- **Model Training**: Train both the unsupervised and reinforcement learning models by running the appropriate training scripts.
- **Evaluation**: Analyze model performance and compare results against a buy-and-hold strategy.

## Debugging

For debugging purposes, the project uses **debugpy**. This is especially useful during model training to inspect intermediate results and verify that the training process is proceeding as expected.

- **Starting Debugging**: Configure your IDE (e.g., VSCode) to attach to the running Docker container where the training script is executing.
- **Breakpoints and Inspection**: Set breakpoints in your training code to inspect variables, monitor data transformations, and evaluate model performance metrics in real time.

## Planned Enhancements

- **Unsupervised Learning Integration**: Fully implement and integrate unsupervised learning methods to automatically identify market trends and regimes.
- **Automated Pipelines**: Develop robust pipelines to automate the end-to-end process—from data fetching and preprocessing to model training and evaluation.
- **Advanced Data Normalization**: Improve normalization techniques to better manage mixed units and enhance learning efficiency.
- **Strategy Optimization**: Enhance the reinforcement learning model to consistently outperform a buy-and-hold strategy.
- **Enhanced Architectures**: Explore advanced architectures such as recurrent or transformer-based policy networks for improved long-term memory and sequence processing.
- **Result Visualizing**: Visualize the results using matplotlib. 

## Future Improvements

- **Market Trend Identification**: Further refine the unsupervised learning techniques to detect market regimes with higher precision.
- **Performance Benchmarking**: Continuously benchmark model performance against traditional investment strategies.
- **User Interface**: Develop a dashboard for real-time monitoring and visualization of model predictions and performance.
