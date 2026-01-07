import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, log_loss

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ======================================================
# 1. PATH CONFIGURATION
# ======================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(CURRENT_DIR, 'train')):
    DATASET_PATH = CURRENT_DIR
elif os.path.exists(os.path.join(CURRENT_DIR, 'UCI HAR Dataset', 'train')):
    DATASET_PATH = os.path.join(CURRENT_DIR, 'UCI HAR Dataset')
else:
    print(" Error (train/Inertial Signals)!")
    exit()

# ======================================================
# STEP 3: PREPROCESSING
# ======================================================
print("\n" + "="*40)
print("--- STEP 3: PREPROCESSING (RAW SIGNALS) ---")
print("="*40)

INPUT_SIGNAL_TYPES = [
    "body_acc_x_", "body_acc_y_", "body_acc_z_",
    "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
    "total_acc_x_", "total_acc_y_", "total_acc_z_"
]

def load_raw_signals(subset):
    path = os.path.join(DATASET_PATH, subset, "Inertial Signals")
    stacked_signals = []

    for signal in INPUT_SIGNAL_TYPES:
        filename = f"{signal}{subset}.txt"
        filepath = os.path.join(path, filename)

        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            exit()

        df = pd.read_csv(filepath, sep=r"\s+", header=None)
        stacked_signals.append(df.values)

    return np.dstack(stacked_signals)

def load_y(subset):
    filepath = os.path.join(DATASET_PATH, subset, f"y_{subset}.txt")
    return pd.read_csv(filepath, sep=r"\s+", header=None)[0].values


# Loading raw data
print("Loading & Reshaping Raw Data...")
X_train = load_raw_signals('train')
X_test = load_raw_signals('test')

y_train_raw = load_y('train')
y_test_raw = load_y('test')

print(f"-> X_train shape: {X_train.shape}")
print(f"-> X_test shape:  {X_test.shape}")

# Label encoding
print("\n[2] Label Encoding...")
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
y_test = le.transform(y_test_raw)

# Scaling
print("\n[3] Scaling Raw Signals...")

scaler = StandardScaler()

def scale_data(data):
    num_samples, num_timesteps, num_channels = data.shape
    data_reshaped = data.reshape(-1, num_channels)
    return data_reshaped, num_samples, num_timesteps, num_channels

X_train_2d, n_s_train, n_t, n_c = scale_data(X_train)
X_train_scaled_2d = scaler.fit_transform(X_train_2d)
X_train_scaled = X_train_scaled_2d.reshape(n_s_train, n_t, n_c)

X_test_2d, n_s_test, _, _ = scale_data(X_test)
X_test_scaled_2d = scaler.transform(X_test_2d)
X_test_scaled = X_test_scaled_2d.reshape(n_s_test, n_t, n_c)

print("-> Scaling Done.")


# ======================================================
# STEP 4: LSTM MODEL
# ======================================================
print("\n" + "="*40)
print("--- STEP 4: LSTM MODEL TRAINING ---")
print("="*40)

n_timesteps, n_features = X_train_scaled.shape[1], X_train_scaled.shape[2]
n_outputs = len(np.unique(y_train))

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(n_timesteps, n_features)),
    BatchNormalization(),
    Dropout(0.4),

    LSTM(64),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(n_outputs, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print(model.summary())

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("best_lstm_model.keras", save_best_only=True)
]

# Train
print("\nTraining LSTM Model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test_scaled, y_test),
    callbacks=callbacks,
    verbose=1
)

# ======================================================
# EVALUATION
# ======================================================
print("\n[Evaluation]")

y_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
loss = log_loss(y_test, y_prob)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Log Loss:  {loss:.4f}")

# Plot curves
def plot_learning_curve(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

plot_learning_curve(history)

print("\n--- All Deep Learning Tasks Completed ---")



# ======================================================
# STEP: SAVE DL SCORES TO JSON
# ======================================================

print("\n💾 Saving DL scores to 'dl_scores.json'...")


lstm_pred_prob = model.predict(X_test_scaled, verbose=0)
lstm_acc = accuracy_score(y_test, np.argmax(lstm_pred_prob, axis=1))
lstm_loss = log_loss(y_test, lstm_pred_prob)

dl_scores = [
    {"Model": "LSTM (Deep Learning)", "Accuracy": round(lstm_acc * 100, 2), "Log Loss": round(lstm_loss, 4)}
]

with open('dl_scores.json', 'w') as f:
    json.dump(dl_scores, f)

print("✅ DL Scores saved!")





























# # ======================================================
# # THE GRAND FINALE: COMPARISON REPORT
# # ======================================================
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# print("\n" + "="*60)
# print("⏳ LSTM Finished. Now benchmarking against ML Models...")
# print("="*60)

# n_samples_train, n_timesteps, n_features = X_train_scaled.shape
# X_train_flat = X_train_scaled.reshape(n_samples_train, -1)

# n_samples_test, _, _ = X_test_scaled.shape
# X_test_flat = X_test_scaled.reshape(n_samples_test, -1)

# #Logistic Regression 
# lr = LogisticRegression(max_iter=1000, solver='liblinear')
# lr.fit(X_train_flat, y_train)
# lr_acc = accuracy_score(y_test, lr.predict(X_test_flat))

# #Random Forest
# rf = RandomForestClassifier(n_estimators=100, max_depth=10)
# rf.fit(X_train_flat, y_train)
# rf_acc = accuracy_score(y_test, rf.predict(X_test_flat))

# # Deep Learning (LSTM)
# lstm_pred = model.predict(X_test_scaled, verbose=0)
# lstm_acc = accuracy_score(y_test, np.argmax(lstm_pred, axis=1))

# # ======================================================
# # FINAL LEADERBOARD
# # ======================================================


# results = {
#     'Model': ['Logistic Regression', 'Random Forest', 'LSTM (Deep Learning)'],
#     'Accuracy': [lr_acc, rf_acc, lstm_acc]
# }

# df_results = pd.DataFrame(results)
# df_results['Accuracy %'] = (df_results['Accuracy'] * 100).round(2)
# df_results = df_results.sort_values(by='Accuracy', ascending=False) # ترتيب للأعلى

# print("\n📊 FINAL LEADERBOARD:")
# print("-" * 40)
# print(df_results[['Model', 'Accuracy %']].to_string(index=False))
# print("-" * 40)


# winner_name = df_results.iloc[0]['Model']
# winner_score = df_results.iloc[0]['Accuracy %']

# print(f"\THE WINNER IS: \033[1m{winner_name}\033[0m with {winner_score}% Accuracy!")
# print("="*60)