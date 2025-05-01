# Kaggle Solana Skill Sprint: Memcoin Graduation Prediction

## Project Overview
This project aims to predict whether a newly minted Solana token ("Memcoin") will **reach 85 SOL liquidity** within its first 100 on-chain blocks. Leveraging on-chain transaction logs and token metadata, we perform extensive feature engineering and ensemble modeling to achieve state-of-the-art performance.

## Data Processing
- **Raw Data Sources**:
  - `train.csv`: Training labels and token identifiers.
  - `dune_token_info.csv` & `token_info_onchain_divers.csv`: On-chain token creation timestamps and metadata.
  - `chunk*.csv`: Behavioral transaction chunks.
- **Processing Steps**:
  1. Load and concatenate all behavioral chunks.
  2. Compute time and slot offsets relative to token mint.
  3. Aggregate per-token statistics (transaction counts, balance trajectories, volume spikes).
  4. Generate windowed features (e.g., SOL volumes in early slots, slope of balance change).
  5. Merge all engineered features with metadata and training labels.

## Feature Engineering
- **Aggregate Metrics**: Count of transactions, unique traders, sum/mean/std of volumes.
- **Windowed Statistics**: SOL volume sums in specified slot and time windows.
- **Trend Features**: Slope of virtual SOL balance.
- **Ratio Features**: Volume-to-wallet ratios, price-change percentages, retention ratios.
- **Spike Detection**: Ratios of max-to-mean volume per slot.

## Modeling
We train three gradient-boosting models with Optuna-powered hyperparameter tuning:
1. **XGBoost** 
2. **LightGBM** 
3. **CatBoost** 

- **Feature Selection**: `SelectFromModel` to reduce to top-K important features per model.
- **Ensembling**: Soft-voting average of per-model probabilities.
- **Validation**: 80/20 train/validation split with early stopping.

## Results
- **Final Validation Log-Loss**: `0.0318`
- **Competition Rank**: Top 15% on leaderboard.

