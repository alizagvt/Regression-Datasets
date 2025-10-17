import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Insurance Regression Explorer", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/alizagvt/Regression-Datasets/d5810552572bad1d3d1af6181e6fdbd28820c9cb/insurance.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df

df = load_data()

st.title("Insurance Charges — Data Exploration & Simple Regression")
st.markdown("This app is adapted from the notebook in the repository. It uses LinearRegression from scikit-learn.")

# Sidebar controls
st.sidebar.header("Options")
show_data = st.sidebar.checkbox("Show raw data", value=True)
show_summary = st.sidebar.checkbox("Show summary stats", value=False)
show_corr = st.sidebar.checkbox("Show correlation heatmap", value=True)
encode_method = st.sidebar.selectbox("Categorical encoding", ["LabelEncode (default)", "OneHotEncode"], index=0)
test_size = st.sidebar.slider("Test set size (%)", 10, 50, 20)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

# Data display
if show_data:
    st.subheader("Dataset (first 10 rows)")
    st.dataframe(df.head(10))

if show_summary:
    st.subheader("Summary statistics")
    st.write(df.describe(include='all'))

st.subheader("Columns")
st.write(list(df.columns))

# Preprocessing options
st.markdown("---")
st.subheader("Feature selection & preprocessing")
all_features = list(df.columns)
default_features = ["age", "bmi", "children"]
features = st.multiselect("Select features (X)", all_features[:-1], default=default_features)
target = st.selectbox("Select target (y)", all_features, index=all_features.index("charges"))

if target in features:
    st.warning("Target was included among features — it will be removed from X automatically.")
    if target in features:
        features = [f for f in features if f != target]

# Prepare X, y
X = df[features].copy()
y = df[target].copy()

# Encode categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
st.write("Categorical features detected:", cat_cols if cat_cols else "None")

if encode_method == "LabelEncode (default)":
    le_map = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        le_map[c] = le
else:  # OneHotEncode
    X = pd.get_dummies(X, drop_first=True)

# Correlation heatmap
if show_corr:
    st.subheader("Correlation heatmap (features + target)")
    viz_df = pd.concat([X, y.reset_index(drop=True)], axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(viz_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Train/test split and model
st.markdown("---")
st.subheader("Train Linear Regression")

test_fraction = test_size / 100.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=int(random_state))

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.write(f"Trained LinearRegression on {len(X_train)} samples; test set {len(X_test)} samples")
st.metric("RMSE (test)", f"{rmse:.2f}")
st.metric("R^2 (test)", f"{r2:.3f}")

# Residuals plot
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, alpha=0.6)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.set_title("Actual vs Predicted")
st.pyplot(fig2)

# Feature coefficients (if linear)
if hasattr(model, 'coef_'):
    st.subheader("Model coefficients")
    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": model.coef_
    }).sort_values("coefficient", key=abs, ascending=False)
    st.dataframe(coef_df.reset_index(drop=True))

# Prediction UI
st.markdown("---")
st.subheader("Make a prediction (single row)")
input_vals = {}
for col in X.columns:
    if np.issubdtype(X[col].dtype, np.number):
        input_vals[col] = st.number_input(f"{col}", value=float(X[col].median()))
    else:
        # treat non-numeric as category
        choices = df[col].astype(str).unique().tolist()
        input_vals[col] = st.selectbox(f"{col}", choices, index=0)

if st.button("Predict"):
    sample = pd.DataFrame([input_vals])
    # if one-hot encoding used earlier, align columns
    if encode_method == "OneHotEncode":
        sample = pd.get_dummies(sample)
        # align columns with training data
        sample = sample.reindex(columns=X.columns, fill_value=0)
    else:
        # label-encoded mapping: we encoded original X earlier with labelencoder for training; need to reuse same labels
        for c in cat_cols:
            # if value unknown to label encoder, attempt to map or add 0
            try:
                le = le_map[c]
                sample[c] = le.transform(sample[c].astype(str))
            except Exception:
                # unseen category -> fallback to 0
                sample[c] = 0
    pred = model.predict(sample)[0]
    st.success(f"Predicted {target}: {pred:.2f}")

st.markdown("---")
st.caption(f"Data source: {DATA_URL}")
