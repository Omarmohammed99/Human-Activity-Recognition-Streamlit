import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, log_loss
from sklearn.model_selection import learning_curve




# ======================================================
# 1. PATH CONFIGURATION 
# ======================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(CURRENT_DIR, 'train')):
    
    DATASET_PATH = CURRENT_DIR
    print(f"Path Detected: Inside Dataset folder directly.")
elif os.path.exists(os.path.join(CURRENT_DIR, 'UCI HAR Dataset', 'train')):
    
    DATASET_PATH = os.path.join(CURRENT_DIR, 'UCI HAR Dataset')
    print(f"Path Detected: Inside subdirectory 'UCI HAR Dataset'.")
else:
    print("\nError:(train)!")
    print(f"not found here {CURRENT_DIR}")
    exit()

def load_file(subset, filename):
    filepath = os.path.join(DATASET_PATH, subset, filename)
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        exit()
    return pd.read_csv(filepath, sep=r"\s+", header=None).values

# ======================================================
# STEP 1: PREPROCESSING
# ======================================================
print("\n" + "="*40)
print("--- STEP 1: PREPROCESSING ---")
print("="*40)

print("Loading Data...")
X_train = load_file('train', 'X_train.txt')
y_train = load_file('train', 'y_train.txt').flatten()
X_test = load_file('test', 'X_test.txt')
y_test = load_file('test', 'y_test.txt').flatten()

print(f"   -> Train shape: {X_train.shape}")
print(f"   -> Test shape:  {X_test.shape}")

print("\n Label Encoding...")
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print("\n Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n PCA Reduction...")
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"   -> Features reduced from {X_train.shape[1]} to {X_train_pca.shape[1]}")

# ======================================================
# HELPER: LEARNING CURVE PLOTTER
# ======================================================
def plot_complexity_curve(estimator, X, y, title):
    print(f"   -> Plotting Learning Curve for {title}...")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=3, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
    )
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="g", label="Validation Score")
    plt.title(f"Learning Curve: {title}")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# ======================================================
# STEP 2: MODELING
# ======================================================
print("\n--- STEP 2: MODELING (Anti-Overfitting) ---")

# --- Model 1: Logistic Regression ---

print("\n[Model 1] Logistic Regression (Regularized)")
lr = LogisticRegression(C=2, solver='liblinear', max_iter=5000, random_state=42)
lr.fit(X_train_pca, y_train)

y_pred = lr.predict(X_test_pca)
y_prob = lr.predict_proba(X_test_pca)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Log Loss:  {log_loss(y_test, y_prob):.4f}")
plot_complexity_curve(lr, X_train_pca, y_train, "Logistic Regression")

# --- Model 2: Random Forest ---
print("\nRandom Forest")
rf = RandomForestClassifier(
    n_estimators=200, 
    max_depth=8,            
    min_samples_leaf=5,     
    max_features='sqrt', 
    
    random_state=42, 
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)
y_prob = rf.predict_proba(X_test_scaled)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Log Loss:  {log_loss(y_test, y_prob):.4f}")
plot_complexity_curve(rf, X_train_scaled, y_train, "Random Forest (Pruned)")

print("\nDone")


# ======================================================
# STEP: SAVE ML SCORES TO JSON (FINAL FIX)
# ======================================================


print("\n💾 Saving ML scores to 'ml_scores.json'...")

# 1. Logistic Regression 
try:
    lr_pred = lr.predict(X_test_pca)
    lr_prob = lr.predict_proba(X_test_pca)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_loss = log_loss(y_test, lr_prob)
except:
    lr_acc, lr_loss = 0, 0

# 2. Random Forest 
try:
    rf_pred = rf.predict(X_test_scaled)    
    rf_prob = rf.predict_proba(X_test_scaled) 
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_loss = log_loss(y_test, rf_prob)
except Exception as e:
    
    print(f"⚠️ Warning: {e}. Trying raw X_test...")
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_loss = 0


ml_scores = [
    {"Model": "Logistic Regression", "Accuracy": round(lr_acc * 100, 2), "Log Loss": round(lr_loss, 4)},
    {"Model": "Random Forest", "Accuracy": round(rf_acc * 100, 2), "Log Loss": round(rf_loss, 4)}
]

with open('ml_scores.json', 'w') as f:
    json.dump(ml_scores, f)

print("✅ ML Scores saved successfully!")

