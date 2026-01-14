import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Regression Models
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# DL Models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = "top100_stocks_final.csv"
RESULTS_FILE = "regression_results.csv"
PLOTS_PREFIX = "regression_comparison"

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("\n" + "="*80)
print("STOCK RETURN PREDICTION - REGRESSION ANALYSIS")
print("Predicting Actual Return Magnitudes (not just direction)")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n STEP 1: Loading and Preparing Data")
print("="*80)

# Load data
df = pd.read_csv(INPUT_FILE)
print(f" Loaded {len(df):,} rows, {len(df.columns)} columns")

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by ticker and date
df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

# Define feature columns (exclude metadata and targets)
exclude_cols = ['Date', 'Ticker', 'Region', 'Target_Binary', 'Target_MultiClass', 
                'Next_Day_Return', 'Next_Day_Price', 'Return_5D_Ahead', 'Return_10D_Ahead']

feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\n Data Summary:")
print(f"   Total features: {len(feature_cols)}")
print(f"   Target variable: Next_Day_Return (continuous values)")
print(f"   Task: Predict actual percentage return for tomorrow")

# Check for missing values
missing = df[feature_cols + ['Next_Day_Return']].isnull().sum().sum()
print(f"   Missing values: {missing}")

if missing > 0:
    print("     Dropping rows with missing values...")
    df = df.dropna(subset=feature_cols + ['Next_Day_Return'])
    print(f"   Remaining rows: {len(df):,}")

# Prepare X and y
X = df[feature_cols].values
y_regression = df['Next_Day_Return'].values  # Actual returns (continuous)

# Also prepare binary target for directional accuracy
y_direction = (df['Next_Day_Return'] > 0).astype(int).values

print(f"\n Dataset Shape:")
print(f"   X (features): {X.shape}")
print(f"   y (returns):  {y_regression.shape}")

# Return statistics
print(f"\n Target Variable Statistics (Next_Day_Return):")
print(f"   Mean:   {y_regression.mean():.6f} ({y_regression.mean()*100:.4f}%)")
print(f"   Median: {np.median(y_regression):.6f}")
print(f"   Std:    {np.std(y_regression):.6f}")
print(f"   Min:    {y_regression.min():.6f} ({y_regression.min()*100:.2f}%)")
print(f"   Max:    {y_regression.max():.6f} ({y_regression.max()*100:.2f}%)")

# Distribution
positive_days = (y_regression > 0).sum()
negative_days = (y_regression < 0).sum()
zero_days = (y_regression == 0).sum()
print(f"\n   Positive returns: {positive_days:,} ({positive_days/len(y_regression)*100:.1f}%)")
print(f"   Negative returns: {negative_days:,} ({negative_days/len(y_regression)*100:.1f}%)")
print(f"   Zero returns:     {zero_days:,} ({zero_days/len(y_regression)*100:.1f}%)")

# ============================================================================
# STEP 2: TRAIN-TEST SPLIT (TEMPORAL)
# ============================================================================

print("\n STEP 2: Train-Test Split (Temporal)")
print("="*80)

# Use temporal split (80-20) - don't shuffle to preserve time series nature
split_idx = int(len(X) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_regression[:split_idx], y_regression[split_idx:]
y_dir_train, y_dir_test = y_direction[:split_idx], y_direction[split_idx:]

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set:     {X_test.shape[0]:,} samples")

# Get date ranges
train_dates = df['Date'].iloc[:split_idx]
test_dates = df['Date'].iloc[split_idx:]

print(f"\nTrain period: {train_dates.min().date()} to {train_dates.max().date()}")
print(f"Test period:  {test_dates.min().date()} to {test_dates.max().date()}")

# ============================================================================
# STEP 3: FEATURE SCALING
# ============================================================================

print("\n STEP 3: Feature Scaling (StandardScaler)")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f" Features scaled to mean=0, std=1")

# ============================================================================
# STEP 4: TRAIN REGRESSION MODELS
# ============================================================================

print("\n STEP 4: Training Regression Models")
print("="*80)

results = []

def directional_accuracy(y_true, y_pred):
    """Calculate percentage of predictions with correct direction"""
    return np.mean(np.sign(y_true) == np.sign(y_pred))

# -----------------
# 4.1: Ridge Regression
# -----------------
print("\n1️  Ridge Regression")
print("-" * 40)

ridge_model = Ridge(random_state=RANDOM_STATE, alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

y_pred_ridge = ridge_model.predict(X_test_scaled)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)
dir_acc_ridge = directional_accuracy(y_test, y_pred_ridge)

print(f"   MAE (Mean Absolute Error): {mae_ridge:.6f} ({mae_ridge*100:.4f}%)")
print(f"   RMSE (Root Mean Squared Error): {rmse_ridge:.6f}")
print(f"   R² Score: {r2_ridge:.6f}")
print(f"   Directional Accuracy: {dir_acc_ridge:.4f} ({dir_acc_ridge*100:.2f}%)")

results.append({
    'Model': 'Ridge Regression',
    'MAE': mae_ridge,
    'RMSE': rmse_ridge,
    'R²': r2_ridge,
    'Directional_Accuracy': dir_acc_ridge
})

# -----------------
# 4.2: KNN Regressor
# -----------------
print("\n2️  K-Nearest Neighbors Regressor")
print("-" * 40)

knn_model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)

mae_knn = mean_absolute_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
r2_knn = r2_score(y_test, y_pred_knn)
dir_acc_knn = directional_accuracy(y_test, y_pred_knn)

print(f"   MAE: {mae_knn:.6f} ({mae_knn*100:.4f}%)")
print(f"   RMSE: {rmse_knn:.6f}")
print(f"   R² Score: {r2_knn:.6f}")
print(f"   Directional Accuracy: {dir_acc_knn:.4f} ({dir_acc_knn*100:.2f}%)")

results.append({
    'Model': 'KNN',
    'MAE': mae_knn,
    'RMSE': rmse_knn,
    'R²': r2_knn,
    'Directional_Accuracy': dir_acc_knn
})

# -----------------
# 4.3: Random Forest Regressor
# -----------------
print("\n3️  Random Forest Regressor")
print("-" * 40)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                 random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
dir_acc_rf = directional_accuracy(y_test, y_pred_rf)

print(f"   MAE: {mae_rf:.6f} ({mae_rf*100:.4f}%)")
print(f"   RMSE: {rmse_rf:.6f}")
print(f"   R² Score: {r2_rf:.6f}")
print(f"   Directional Accuracy: {dir_acc_rf:.4f} ({dir_acc_rf*100:.2f}%)")

results.append({
    'Model': 'Random Forest',
    'MAE': mae_rf,
    'RMSE': rmse_rf,
    'R²': r2_rf,
    'Directional_Accuracy': dir_acc_rf
})

# -----------------
# 4.4: XGBoost Regressor
# -----------------
print("\n4  XGBoost Regressor")
print("-" * 40)

xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                         random_state=RANDOM_STATE, n_jobs=-1)
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
dir_acc_xgb = directional_accuracy(y_test, y_pred_xgb)

print(f" MAE: {mae_xgb:.6f} ({mae_xgb*100:.4f}%)")
print(f"   RMSE: {rmse_xgb:.6f}")
print(f"   R² Score: {r2_xgb:.6f}")
print(f"   Directional Accuracy: {dir_acc_xgb:.4f} ({dir_acc_xgb*100:.2f}%)")

results.append({
    'Model': 'XGBoost',
    'MAE': mae_xgb,
    'RMSE': rmse_xgb,
    'R²': r2_xgb,
    'Directional_Accuracy': dir_acc_xgb
})

# ============================================================================
# STEP 5: TRAIN LSTM MODEL (REGRESSION)
# ============================================================================

print("\n STEP 5: Training LSTM Model (Regression)")
print("="*80)

# Prepare sequences for LSTM
SEQ_LENGTH = 60

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

print(f"\n Creating sequences (sequence length: {SEQ_LENGTH})")

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQ_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, SEQ_LENGTH)

print(f"   Train sequences: {X_train_seq.shape}")
print(f"   Test sequences:  {X_test_seq.shape}")

# Build LSTM model for regression
print(f"\n  Building LSTM architecture...")

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, X_train_scaled.shape[1])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)  # No activation for regression (linear output)
])

lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',  # Mean Squared Error for regression
    metrics=['mae']  # Mean Absolute Error
)

print(f" LSTM Model Summary:")
lstm_model.summary()

# Train LSTM
print(f"\n Training LSTM...")

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate LSTM
print(f"\n Evaluating LSTM...")

y_pred_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()

mae_lstm = mean_absolute_error(y_test_seq, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_seq, y_pred_lstm))
r2_lstm = r2_score(y_test_seq, y_pred_lstm)
dir_acc_lstm = directional_accuracy(y_test_seq, y_pred_lstm)

print(f"   MAE: {mae_lstm:.6f} ({mae_lstm*100:.4f}%)")
print(f"   RMSE: {rmse_lstm:.6f}")
print(f"   R² Score: {r2_lstm:.6f}")
print(f"   Directional Accuracy: {dir_acc_lstm:.4f} ({dir_acc_lstm*100:.2f}%)")

results.append({
    'Model': 'LSTM',
    'MAE': mae_lstm,
    'RMSE': rmse_lstm,
    'R²': r2_lstm,
    'Directional_Accuracy': dir_acc_lstm
})

# ============================================================================
# STEP 6: RESULTS COMPARISON
# ============================================================================

print("\n STEP 6: Model Comparison Results")
print("="*80)

results_df = pd.DataFrame(results)
# Sort by Directional Accuracy (most important for trading)
results_df = results_df.sort_values('Directional_Accuracy', ascending=False).reset_index(drop=True)

print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv(RESULTS_FILE, index=False)
print(f"\n Results saved to: {RESULTS_FILE}")

# Find best model
best_model = results_df.iloc[0]
print(f"\n BEST MODEL (by Directional Accuracy): {best_model['Model']}")
print(f"   Directional Accuracy: {best_model['Directional_Accuracy']:.4f}")
print(f"   MAE: {best_model['MAE']:.6f}")
print(f"   R² Score: {best_model['R²']:.6f}")

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================

print("\n STEP 7: Creating Visualizations")
print("="*80)

# Set style
sns.set_style("whitegrid")

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))

# 1. Directional Accuracy Comparison
ax1 = plt.subplot(2, 3, 1)
sns.barplot(data=results_df, x='Model', y='Directional_Accuracy', palette='viridis', ax=ax1)
ax1.set_title('Directional Accuracy by Model', fontsize=14, fontweight='bold')
ax1.set_ylabel('Directional Accuracy')
ax1.set_ylim(0.45, 0.65)
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(results_df['Directional_Accuracy']):
    ax1.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# 2. MAE Comparison (Lower is better)
ax2 = plt.subplot(2, 3, 2)
sns.barplot(data=results_df, x='Model', y='MAE', palette='magma', ax=ax2)
ax2.set_title('Mean Absolute Error (Lower = Better)', fontsize=14, fontweight='bold')
ax2.set_ylabel('MAE')
ax2.tick_params(axis='x', rotation=45)
for i, v in enumerate(results_df['MAE']):
    ax2.text(i, v + 0.0002, f'{v:.6f}', ha='center', va='bottom', fontsize=9)

# 3. R² Score Comparison
ax3 = plt.subplot(2, 3, 3)
sns.barplot(data=results_df, x='Model', y='R²', palette='coolwarm', ax=ax3)
ax3.set_title('R² Score (Higher = Better)', fontsize=14, fontweight='bold')
ax3.set_ylabel('R² Score')
ax3.tick_params(axis='x', rotation=45)
for i, v in enumerate(results_df['R²']):
    ax3.text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

# 4. Actual vs Predicted (XGBoost - best model)
ax4 = plt.subplot(2, 3, 4)
sample_size = min(1000, len(y_test))
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
ax4.scatter(y_test[sample_indices], y_pred_xgb[sample_indices], alpha=0.5, s=10)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax4.set_xlabel('Actual Returns', fontsize=12)
ax4.set_ylabel('Predicted Returns', fontsize=12)
ax4.set_title(f'Actual vs Predicted (XGBoost)\nR²={r2_xgb:.4f}', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Prediction Error Distribution (XGBoost)
ax5 = plt.subplot(2, 3, 5)
errors = y_test - y_pred_xgb
ax5.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax5.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
ax5.set_xlabel('Prediction Error', fontsize=12)
ax5.set_ylabel('Frequency', fontsize=12)
ax5.set_title(f'Error Distribution (XGBoost)\nMean Error: {errors.mean():.6f}', 
              fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. All Metrics Heatmap
ax6 = plt.subplot(2, 3, 6)
metrics_df = results_df.set_index('Model')[['MAE', 'RMSE', 'R²', 'Directional_Accuracy']]
sns.heatmap(metrics_df.T, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax6, 
            cbar_kws={'label': 'Score'})
ax6.set_title('All Metrics Heatmap', fontsize=14, fontweight='bold')
ax6.set_ylabel('')

plt.tight_layout()
plt.savefig(f'{PLOTS_PREFIX}_overview.png', dpi=300, bbox_inches='tight')
print(f" Saved: {PLOTS_PREFIX}_overview.png")

# Additional plot: Prediction samples over time
fig2, ax = plt.subplots(figsize=(16, 6))
sample_period = slice(0, 500)  # First 500 test samples
ax.plot(y_test[sample_period], label='Actual Returns', alpha=0.7, linewidth=1.5)
ax.plot(y_pred_xgb[sample_period], label='Predicted Returns (XGBoost)', 
        alpha=0.7, linewidth=1.5, linestyle='--')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_xlabel('Sample Index', fontsize=12)
ax.set_ylabel('Return', fontsize=12)
ax.set_title('Actual vs Predicted Returns Over Time (First 500 Test Samples)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_PREFIX}_predictions_timeline.png', dpi=300, bbox_inches='tight')
print(f" Saved: {PLOTS_PREFIX}_predictions_timeline.png")

plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("REGRESSION ANALYSIS COMPLETE!")
print("="*80)

print(f"\n Summary:")
print(f"   Models trained: {len(results)}")
print(f"   Best model: {best_model['Model']}")
print(f"   Best directional accuracy: {best_model['Directional_Accuracy']:.4f}")
print(f"   Best MAE: {best_model['MAE']:.6f}")

print(f"\n Output Files:")
print(f"   - {RESULTS_FILE}")
print(f"   - {PLOTS_PREFIX}_overview.png")
print(f"   - {PLOTS_PREFIX}_predictions_timeline.png")

print("\n Key Insights:")
print(f"   - Directional Accuracy: {best_model['Directional_Accuracy']*100:.2f}% (similar to classification)")
print(f"   - Average prediction error: {best_model['MAE']*100:.4f}%")
print(f"   - R² Score: {best_model['R²']:.4f} (low is normal for stock returns)")
print(f"   - Regression provides return MAGNITUDE for position sizing")

print("\n Trading Application:")
print("   Use regression predictions for:")
print("   1. Position sizing (larger positions for larger predicted moves)")
print("   2. Risk management (expected return vs potential loss)")
print("   3. Portfolio optimization (weight stocks by expected returns)")
print("   4. Setting price targets and stop losses")

print("\n Study complete! Check output files for detailed results.")
print("="*80 + "\n")
