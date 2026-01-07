import streamlit as st
import json
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. SETUP
# ==========================================
st.set_page_config(
    page_title="HAR Activity Predictor",
    page_icon="🏃‍♂️",
    layout="wide"
)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(current_dir, 'train')):
        return current_dir
    elif os.path.exists(os.path.join(current_dir, 'UCI HAR Dataset', 'train')):
        return os.path.join(current_dir, 'UCI HAR Dataset')
    return None

@st.cache_resource
def train_model():
    base_path = get_dataset_path()
    if not base_path:
        return None, None, None, None

    # Load Data
    try:
        X_train = pd.read_csv(os.path.join(base_path, 'train', 'X_train.txt'), sep=r"\s+", header=None).values
        y_train = pd.read_csv(os.path.join(base_path, 'train', 'y_train.txt'), sep=r"\s+", header=None).values.flatten()
    except:
        return None, None, None, None
    
    # Preprocessing
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # PCA (Optional)
    pca = PCA(n_components=3, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Training
    model = LogisticRegression(C=10, solver='liblinear', random_state=42)
    model.fit(X_train_pca, y_train_enc)
    
    return model, scaler, pca, le

# Train Model
model, scaler, pca, le = train_model()

# Activity Map (using original dataset classes)
activity_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}

# ==========================================
# 3. UI LAYOUT
# ==========================================
st.title("🏃‍♂️ Human Activity Recognition App")
st.markdown("Predicts activity based on sensor data simulation.")
st.divider()

if model is None:
    st.error("❌ Error: Dataset not found! Make sure 'UCI HAR Dataset' is next to app.py")
    st.stop()

# --- Sidebar ---
st.sidebar.header("⚙️ User Input")
input_type = st.sidebar.radio("Input Method:", ["Manual Sliders", "Upload CSV"])

input_data = None

if input_type == "Manual Sliders":
    st.sidebar.info("Simulate sensor data using PCA components.")
    pc1 = st.sidebar.slider("Component 1", -10.0, 10.0, 0.0)
    pc2 = st.sidebar.slider("Component 2", -10.0, 10.0, 0.0)
    pc3 = st.sidebar.slider("Component 3", -10.0, 10.0, 0.0)
    input_data = np.array([[pc1, pc2, pc3]])

elif input_type == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload X_test snippet", type=["txt", "csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, sep=r"\s+", header=None)
            raw_samples = df.values
            scaled_samples = scaler.transform(raw_samples)
            input_data = pca.transform(scaled_samples)
            st.sidebar.success(f"File loaded! {input_data.shape[0]} rows ready for prediction.")
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# 4. PREDICTION & DISPLAY
# ==========================================
col1, col2 = st.columns([1, 2])

if input_data is not None:
    predictions_idx = model.predict(input_data)
    predictions_prob = model.predict_proba(input_data)

    # Convert to original activity labels safely
    predictions_labels = le.inverse_transform(predictions_idx)
    predicted_activities = [activity_map.get(lbl, "UNKNOWN") for lbl in predictions_labels]

    # --- Column 1: Result ---
    with col1:
        st.subheader("🤖 Predictions")
        if len(predicted_activities) == 1:
            st.success(f"**{predicted_activities[0]}**")
        else:
            st.dataframe(pd.DataFrame({
                "Predicted Activity": predicted_activities
            }))

    # --- Column 2: Chart ---
    with col2:
        st.subheader("📊 Confidence")
        if len(predicted_activities) == 1:
            chart_data = pd.DataFrame({
                "Activity": list(activity_map.values()),
                "Probability": predictions_prob[0]
            })
            st.bar_chart(chart_data.set_index("Activity"))
        else:
            # Multiple rows: show mean probability per class
            chart_data = pd.DataFrame(predictions_prob, columns=activity_map.values())
            st.dataframe(chart_data)

else:
    st.info("👈 Please use the sidebar to start.")


# ==========================================
# 5. MODEL COMPARISON
# ==========================================


st.divider()
st.header("🏆 Model Comparison Benchmark")
st.markdown("Comparing Classical ML models (on Engineered Features) vs Deep Learning (on Raw Data).")


all_scores = []


if os.path.exists('ml_scores.json'):
    with open('ml_scores.json', 'r') as f:
        all_scores.extend(json.load(f))


if os.path.exists('dl_scores.json'):
    with open('dl_scores.json', 'r') as f:
        all_scores.extend(json.load(f)) 


if all_scores:
    df_compare = pd.DataFrame(all_scores)
    
  
    df_compare = df_compare.sort_values(by="Accuracy", ascending=False)

    col_table, col_chart = st.columns([1, 2])

    with col_table:
        st.subheader("📋 Final Results")
        # تلوين أعلى قيمة بالأخضر
        st.dataframe(df_compare.style.highlight_max(axis=0, subset=['Accuracy'], color='lightgreen'))

    with col_chart:
        st.subheader("📊 Accuracy Chart")
        st.bar_chart(df_compare.set_index("Model")['Accuracy'], color="#4CAF50")
        
 
    best_model = df_compare.iloc[0]
    st.success(f"🏅 **Overall Winner:** {best_model['Model']} ({best_model['Accuracy']}%)")

else:
    st.warning("⚠️ No result files found. Please run 'ML.py' and 'DL.py' first.")
