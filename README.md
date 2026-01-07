# Stock Market Prediction Using Machine Learning and Deep Learning: A Comparative Analysis

A comprehensive comparative study of machine learning and deep learning approaches for stock market prediction, implementing rigorous temporal validation to achieve realistic performance benchmarks.

Key Results

Best Performance: KNN achieved 55.63% directional accuracy (highest)
Deep Learning: LSTM networks achieved competitive 54.91% accuracy
Ensemble Methods: XGBoost (54.68%) and Random Forest (52.73%) showed modest performance
Surprising Finding: Instance-based learning (KNN) outperformed ensemble methods
Realistic Benchmarks: Confirmed 52-56% as realistic accuracy expectations under proper validation
Feature Importance: Local pattern recognition proved most effective for financial time series

Repository Structure

stock-market-prediction/
├── build_dataset.py
├── data_check_final.py
├── data_quality_report.txt
├── dataset_cleaning.py
├── fixing_negatives.py
├── model_comparison_regression.py
├── model_comparison.py
├── top100_stocks_cleaned.csv
├── top100_stocks_features.csv
├── top 100_stocks_final.csv
├── top100_stocks_raw.csv
├── requirements.txt
└── README.md

Dataset Information
Stock Universe (100 Stocks)

United States (35): AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, BAC, KO, PG, GE, etc.
Europe (20): SAP, ASML, LVMH, Airbus, Nestlé, Siemens, etc.
United Kingdom (5): Unilever, HSBC, BP, Shell, Vodafone
India (15): Reliance, TCS, HDFC, Infosys, ICICI, etc.
China/Hong Kong (15): Tencent, Alibaba, TSM, etc.
Japan (8): Toyota, Sony, SoftBank, Nintendo, etc.
Other (2): Samsung, Commonwealth Bank of Australia

Data Specifications

Time Period: January 2000 - December 2025 (25 years)
Total Records: 577,234 daily observations
Features: 90+ engineered technical indicators
Temporal Split: 80% training (2000-2019), 20% testing (2020-2025)

Dataset Access
Due to large file sizes (~2.5GB), the dataset is hosted on OneDrive:
Download Link: https://mydbs-my.sharepoint.com/:f:/g/personal/20063643_mydbs_ie/IgAPkiyK0OmxQqU5ef-xZ4kgAYasJSqt4BN3VdzlYjADk5A?e=6gAHUv


Model Implementations

K-Nearest Neighbors (KNN) - Best Performer

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    metric='euclidean',
    n_jobs=-1
)

Accuracy: 55.63%
Training Time: ~2 minutes
Prediction Time: ~0.8 seconds

Long Short-Term Memory (LSTM)

import tensorflow as tf
from tensorflow.keras import Sequential, Dense, LSTM, Dropout

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, n_features)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1, activation='sigmoid')
])

Accuracy: 54.91%
Training Time: ~38 minutes
Prediction Time: ~0.5 seconds

XGBoost

import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    random_state=42
)

Accuracy: 54.68%
Training Time: ~6.8 minutes
Prediction Time: ~0.12 seconds

Feature Engineering
90+ Technical Indicators Organized by Category:
Price Features (25 features)

Daily Returns, Log Returns, Lagged Returns (1,2,3,5,10 days)
Simple Moving Averages (5,10,20,50,100,200 days)
Exponential Moving Averages (5,10,20,50 days)
Price-to-Moving-Average Ratios
Momentum indicators (5,10,20 days)

Volatility Features (15 features)

Rolling Volatility (5,10,20,50 days)
Average True Range (ATR)
High-Low Range
Volatility Ratios

Volume Features (10 features)

Volume Moving Averages
Volume Ratios
Price-Volume Confirmation Signals

Technical Indicators (30 features)

RSI (14 periods)
MACD (12,26,9 parameters)
Bollinger Bands (width, position)
Stochastic Oscillator (%K, %D)
Williams %R

Time Features (10+ features)

Day of Week, Month, Quarter
Calendar Effects
Market Regime Indicators


Bias Prevention

No Random Splits: Strict chronological order maintained
No Data Leakage: Future information never used in training
Cross-Validation: Time-aware splitting procedures
Parameter Selection: Validation set separate from test set


Performance Limitations

Modest Returns: ~5.6% edge above random (realistic expectation)
Transaction Costs: Real trading involves costs that significantly impact profitability
Market Impact: Large-scale implementation may reduce effectiveness
Regime Dependency: Performance varies significantly across market conditions

Risk Considerations

Past Performance: Does not guarantee future results
Model Limitations: All models can fail during unprecedented market events
Overfitting Risk: Despite precautions, models may not generalize perfectly
Regulatory Changes: Market structure changes may affect model validity


Acknowledgments

Data Source: Yahoo Finance API (yfinance library)
Libraries: scikit-learn, TensorFlow, XGBoost, pandas, numpy
Inspiration: Academic research in computational finance and market efficiency

Contact

Author: Priyanka Diwakar Baral
Email: 20063643@mybds.ie
Institution: Dublin Business School

