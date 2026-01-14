import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve)

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# DL Models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = "top100_stocks_final.csv"
RESULTS_FILE = "model_comparison_results.csv"
PLOTS_PREFIX = "model_comparison"

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("\n" + "="*80)
print("STOCK PRICE PREDICTION - ML/DL COMPARISON STUDY")
print("="*80)


# STEP 1: LOAD AND PREPARE DATA


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

print(f"\n Feature Engineering Summary:")
print(f"   Total features: {len(feature_cols)}")
print(f"   Target variable: Target_Binary (0=Down, 1=Up)")

# Check for missing values
missing = df[feature_cols + ['Target_Binary']].isnull().sum().sum()
print(f"   Missing values: {missing}")

if missing > 0:
    print("     Dropping rows with missing values...")
    df = df.dropna(subset=feature_cols + ['Target_Binary'])
    print(f"    Remaining rows: {len(df):,}")

# Prepare X and y
X = df[feature_cols].values
y = df['Target_Binary'].values

print(f"\n Dataset Shape:")
print(f"   X: {X.shape}")
print(f"   y: {y.shape}")

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print(f"\n Class Distribution:")
for label, count in zip(unique, counts):
    pct = (count / len(y)) * 100
    print(f"   Class {int(label)}: {count:,} ({pct:.1f}%)")


# STEP 2: TRAIN-TEST SPLIT (TEMPORAL)


print("\n STEP 2: Train-Test Split (Temporal)")
print("="*80)

# Use temporal split (80-20) - don't shuffle to preserve time series nature
split_idx = int(len(X) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set:     {X_test.shape[0]:,} samples")

# Get date ranges
train_dates = df['Date'].iloc[:split_idx]
test_dates = df['Date'].iloc[split_idx:]

print(f"\nTrain period: {train_dates.min().date()} to {train_dates.max().date()}")
print(f"Test period:  {test_dates.min().date()} to {test_dates.max().date()}")


# STEP 3: FEATURE SCALING


print("\n STEP 3: Feature Scaling (StandardScaler)")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Features scaled to mean=0, std=1")
print(f"   Sample feature stats after scaling:")
print(f"   Mean: {X_train_scaled[:, 0].mean():.6f}")
print(f"   Std:  {X_train_scaled[:, 0].std():.6f}")


# STEP 4: TRAIN ML MODELS

print("\n STEP 4: Training ML Models")
print("="*80)

results = []


# 4.1: Logistic Regression

print("\n Logistic Regression")
print("-" * 40)

lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
f1_lr = f1_score(y_test, y_pred_lr, zero_division=0)
auc_lr = roc_auc_score(y_test, y_prob_lr)

print(f"   Accuracy:  {acc_lr:.4f}")
print(f"   Precision: {prec_lr:.4f}")
print(f"   Recall:    {rec_lr:.4f}")
print(f"   F1-Score:  {f1_lr:.4f}")
print(f"   ROC-AUC:   {auc_lr:.4f}")

results.append({
    'Model': 'Logistic Regression',
    'Accuracy': acc_lr,
    'Precision': prec_lr,
    'Recall': rec_lr,
    'F1-Score': f1_lr,
    'ROC-AUC': auc_lr
})


# 4.2: KNN

print("\n  K-Nearest Neighbors (K=5)")
print("-" * 40)

knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)

y_pred_knn = knn_model.predict(X_test_scaled)
y_prob_knn = knn_model.predict_proba(X_test_scaled)[:, 1]

acc_knn = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn, zero_division=0)
rec_knn = recall_score(y_test, y_pred_knn, zero_division=0)
f1_knn = f1_score(y_test, y_pred_knn, zero_division=0)
auc_knn = roc_auc_score(y_test, y_prob_knn)

print(f"   Accuracy:  {acc_knn:.4f}")
print(f"   Precision: {prec_knn:.4f}")
print(f"   Recall:    {rec_knn:.4f}")
print(f"   F1-Score:  {f1_knn:.4f}")
print(f"   ROC-AUC:   {auc_knn:.4f}")

results.append({
    'Model': 'KNN',
    'Accuracy': acc_knn,
    'Precision': prec_knn,
    'Recall': rec_knn,
    'F1-Score': f1_knn,
    'ROC-AUC': auc_knn
})


# 4.3: Random Forest

print("\n  Random Forest")
print("-" * 40)

rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, 
                                  max_depth=10, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print(f"   Accuracy:  {acc_rf:.4f}")
print(f"   Precision: {prec_rf:.4f}")
print(f"   Recall:    {rec_rf:.4f}")
print(f"   F1-Score:  {f1_rf:.4f}")
print(f"   ROC-AUC:   {auc_rf:.4f}")

results.append({
    'Model': 'Random Forest',
    'Accuracy': acc_rf,
    'Precision': prec_rf,
    'Recall': rec_rf,
    'F1-Score': f1_rf,
    'ROC-AUC': auc_rf
})


# 4.4: XGBoost

print("\n  XGBoost")
print("-" * 40)

xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                         random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)

y_pred_xgb = xgb_model.predict(X_test_scaled)
y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

acc_xgb = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb, zero_division=0)
rec_xgb = recall_score(y_test, y_pred_xgb, zero_division=0)
f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)

print(f"   Accuracy:  {acc_xgb:.4f}")
print(f"   Precision: {prec_xgb:.4f}")
print(f"   Recall:    {rec_xgb:.4f}")
print(f"   F1-Score:  {f1_xgb:.4f}")
print(f"   ROC-AUC:   {auc_xgb:.4f}")

results.append({
    'Model': 'XGBoost',
    'Accuracy': acc_xgb,
    'Precision': prec_xgb,
    'Recall': rec_xgb,
    'F1-Score': f1_xgb,
    'ROC-AUC': auc_xgb
})


# STEP 5: TRAIN LSTM MODEL

print("\n STEP 5: Training LSTM Model")
print("="*80)

# Prepare sequences for LSTM
SEQ_LENGTH = 60  # Use 60 days of history

def create_sequences(X, y, seq_length):
    """Create sequences for LSTM"""
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

# Build LSTM model
print(f"\n  Building LSTM architecture...")

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, X_train_scaled.shape[1])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
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

y_prob_lstm = lstm_model.predict(X_test_seq).flatten()
y_pred_lstm = (y_prob_lstm > 0.5).astype(int)

acc_lstm = accuracy_score(y_test_seq, y_pred_lstm)
prec_lstm = precision_score(y_test_seq, y_pred_lstm, zero_division=0)
rec_lstm = recall_score(y_test_seq, y_pred_lstm, zero_division=0)
f1_lstm = f1_score(y_test_seq, y_pred_lstm, zero_division=0)
auc_lstm = roc_auc_score(y_test_seq, y_prob_lstm)

print(f"   Accuracy:  {acc_lstm:.4f}")
print(f"   Precision: {prec_lstm:.4f}")
print(f"   Recall:    {rec_lstm:.4f}")
print(f"   F1-Score:  {f1_lstm:.4f}")
print(f"   ROC-AUC:   {auc_lstm:.4f}")

results.append({
    'Model': 'LSTM',
    'Accuracy': acc_lstm,
    'Precision': prec_lstm,
    'Recall': rec_lstm,
    'F1-Score': f1_lstm,
    'ROC-AUC': auc_lstm
})

# Results Comparison

print("\n STEP 6: Model Comparison Results")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)

print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv(RESULTS_FILE, index=False)
print(f"\n Results saved to: {RESULTS_FILE}")

# Find best model
best_model = results_df.iloc[0]
print(f"\n BEST MODEL: {best_model['Model']}")
print(f"   Accuracy: {best_model['Accuracy']:.4f}")
print(f"   F1-Score: {best_model['F1-Score']:.4f}")

# STEP 7: VISUALIZATIONS


print("\n STEP 7: Creating Visualizations")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# 1. Accuracy Comparison
ax1 = plt.subplot(2, 3, 1)
sns.barplot(data=results_df, x='Model', y='Accuracy', palette='viridis', ax=ax1)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0.4, 1.0)
ax1.set_ylabel('Accuracy')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(results_df['Accuracy']):
    ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# 2. F1-Score Comparison
ax2 = plt.subplot(2, 3, 2)
sns.barplot(data=results_df, x='Model', y='F1-Score', palette='magma', ax=ax2)
ax2.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
ax2.set_ylim(0.4, 1.0)
ax2.tick_params(axis='x', rotation=45)
for i, v in enumerate(results_df['F1-Score']):
    ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# 3. ROC-AUC Comparison
ax3 = plt.subplot(2, 3, 3)
sns.barplot(data=results_df, x='Model', y='ROC-AUC', palette='coolwarm', ax=ax3)
ax3.set_title('Model ROC-AUC Comparison', fontsize=14, fontweight='bold')
ax3.set_ylim(0.4, 1.0)
ax3.tick_params(axis='x', rotation=45)
for i, v in enumerate(results_df['ROC-AUC']):
    ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# 4. Precision vs Recall
ax4 = plt.subplot(2, 3, 4)
for idx, row in results_df.iterrows():
    ax4.scatter(row['Recall'], row['Precision'], s=200, alpha=0.6, label=row['Model'])
    ax4.annotate(row['Model'], (row['Recall'], row['Precision']), 
                fontsize=9, ha='center', va='bottom')
ax4.set_xlabel('Recall', fontsize=12)
ax4.set_ylabel('Precision', fontsize=12)
ax4.set_title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

# 5. All Metrics Heatmap
ax5 = plt.subplot(2, 3, 5)
metrics_df = results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
sns.heatmap(metrics_df.T, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax5, cbar_kws={'label': 'Score'})
ax5.set_title('All Metrics Heatmap', fontsize=14, fontweight='bold')
ax5.set_ylabel('')

# 6. ROC Curves (for ML models only - LSTM uses different test set)
ax6 = plt.subplot(2, 3, 6)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

ax6.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.3f})', linewidth=2)
ax6.plot(fpr_knn, tpr_knn, label=f'KNN (AUC={auc_knn:.3f})', linewidth=2)
ax6.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.3f})', linewidth=2)
ax6.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={auc_xgb:.3f})', linewidth=2)
ax6.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax6.set_xlabel('False Positive Rate', fontsize=12)
ax6.set_ylabel('True Positive Rate', fontsize=12)
ax6.set_title('ROC Curves (ML Models)', fontsize=14, fontweight='bold')
ax6.legend(loc='lower right')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_PREFIX}_overview.png', dpi=300, bbox_inches='tight')
print(f" Saved: {PLOTS_PREFIX}_overview.png")

# Confusion Matrices
fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

models_cm = [
    ('Logistic Regression', y_pred_lr, y_test),
    ('KNN', y_pred_knn, y_test),
    ('Random Forest', y_pred_rf, y_test),
    ('XGBoost', y_pred_xgb, y_test),
    ('LSTM', y_pred_lstm, y_test_seq)
]

for idx, (name, y_pred, y_true) in enumerate(models_cm):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

# Hide last subplot if not used
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(f'{PLOTS_PREFIX}_confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f" Saved: {PLOTS_PREFIX}_confusion_matrices.png")

plt.show()

#Final Summary

print("\n" + "="*80)
print("COMPARISON STUDY COMPLETE!")
print("="*80)

print(f"\n Summary:")
print(f"   Total models trained: {len(results)}")
print(f"   Best performing model: {best_model['Model']}")
print(f"   Best accuracy: {best_model['Accuracy']:.4f}")
print(f"   Best F1-Score: {best_model['F1-Score']:.4f}")

print(f"\n Output Files:")
print(f"   - {RESULTS_FILE}")
print(f"   - {PLOTS_PREFIX}_overview.png")
print(f"   - {PLOTS_PREFIX}_confusion_matrices.png")

print("\n Study complete! Check the output files above for detailed results.")
print("="*80 + "\n")
