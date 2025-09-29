# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# --- Supabase auth helper ---
from auth import supabase_client, get_user
sb = supabase_client()

# =========================
# Load trained final model
# =========================
model = joblib.load("final_model.pkl")
df = pd.read_csv("cleaned_vehicle_dataset.csv")

# =========================
# App Layout
# =========================
st.set_page_config(page_title="CO₂ Emissions Predictor", page_icon="🚗", layout="wide")

# ---------- Initialize session_state keys ----------
for key in ["login_email", "login_pwd", "reg_email", "reg_pwd", "login_message", "login_error", "user"]:
    if key not in st.session_state:
        st.session_state[key] = "" if "message" not in key and "error" not in key else None

# ---------- Login callback ----------
def login_callback():
    try:
        sb.auth.sign_in_with_password({
            "email": st.session_state.login_email,
            "password": st.session_state.login_pwd
        })
        st.session_state.login_email = ""
        st.session_state.login_pwd = ""
        st.session_state.login_message = "Logged in successfully!"
        st.session_state.login_error = ""
        st.session_state.user = get_user(sb)  # Refetch user after login
    except Exception as e:
        msg = str(e)
        if "Email not confirmed" in msg:
            st.session_state.login_error = "Your email is not confirmed. Check your inbox (or spam)."
        else:
            st.session_state.login_error = f"Login failed: {e}"
        st.session_state.login_message = ""

# ---------- Register callback ----------
def register_callback():
    try:
        sb.auth.sign_up({
            "email": st.session_state.reg_email,
            "password": st.session_state.reg_pwd
        })
        st.session_state.reg_email = ""
        st.session_state.reg_pwd = ""
        st.session_state.login_message = "Account created. Check your email if confirmation is enabled."
        st.session_state.login_error = ""
    except Exception as e:
        st.session_state.login_error = f"Signup failed: {e}"
        st.session_state.login_message = ""

# ---------- Logout callback ----------
def logout_callback():
    try:
        sb.auth.sign_out()
    except Exception:
        pass
    for k in ["login_email", "login_pwd", "reg_email", "reg_pwd"]:
        st.session_state[k] = ""
    st.session_state.user = None
    st.session_state.login_message = ""
    st.session_state.login_error = ""

# ---------- Authentication ----------
user = st.session_state.get("user", get_user(sb))

if not user:
    st.sidebar.title("Account")
    login_tab, register_tab = st.sidebar.tabs(["🔐 Login", "🆕 Register"])

    # ---------------- Login Tab ----------------
    with login_tab:
        st.text_input("Email", key="login_email", value=st.session_state.login_email)
        st.text_input("Password", type="password", key="login_pwd", value=st.session_state.login_pwd)
        st.button("Login", on_click=login_callback)
        # Display messages
        if st.session_state.login_message:
            st.success(st.session_state.login_message)
        if st.session_state.login_error:
            st.error(st.session_state.login_error)

    # ---------------- Register Tab ----------------
    with register_tab:
        st.text_input("Email (new)", key="reg_email", value=st.session_state.reg_email)
        st.text_input("Password (new)", type="password", key="reg_pwd", value=st.session_state.reg_pwd)
        st.button("Create account", on_click=register_callback)
        if st.session_state.login_message:
            st.success(st.session_state.login_message)
        if st.session_state.login_error:
            st.error(st.session_state.login_error)

    st.stop()  # Stop main app until logged in

# ---------- Sidebar: Logged-in User ----------
st.sidebar.success(f"Signed in as {user.email}")
st.sidebar.button("Logout", on_click=logout_callback)

# ---------- Main App ----------
st.title("🚗 Vehicle CO₂ Emissions Predictor")
st.markdown("Predict vehicle **CO₂ emissions (g/km)** based on specifications using a Multiple Linear Regression model.")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("🔧 Enter Vehicle Specifications")
engine_size = st.sidebar.number_input("Engine size (L)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)
cylinders = st.sidebar.number_input("Number of Cylinders", min_value=3, max_value=12, step=1, value=4)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Ethanol", "Natural Gas", "Premium Petrol"])
combined_l_100km = st.sidebar.number_input("Combined (L/100 km)", min_value=2.0, max_value=25.0, step=0.1, value=8.5)

# ---------------- Prediction ----------------
if st.sidebar.button("Predict"):
    input_data = pd.DataFrame({
        "Engine size (L)": [engine_size],
        "Cylinders": [cylinders],
        "Fuel type": [fuel_type],
        "Combined (L/100 km)": [combined_l_100km]
    })
    prediction = model.predict(input_data)[0]

    # Results
    st.subheader("🔮 Predicted CO₂ Emissions")
    st.metric("CO₂ emissions (g/km)", f"{prediction:.2f}")

    avg_emission = df["CO2 emissions (g/km)"].mean()
    X = df.drop(columns=["CO2 emissions (g/km)", "CO2 rating", "Smog rating"], errors="ignore")
    y = df["CO2 emissions (g/km)"]
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    st.write("---")
    st.subheader("📊 Model Performance")
    st.write(f"**R² Score:** {r2:.4f}")
    st.write(f"**RMSE:** {rmse:.2f} g/km")
    st.write(f"**MAE:** {mae:.2f} g/km")

    # Graphs
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 📈 Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(5,4))
        sns.scatterplot(x=y, y=y_pred, alpha=0.6, ax=ax)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        ax.set_xlabel("Actual CO₂ emissions (g/km)")
        ax.set_ylabel("Predicted CO₂ emissions (g/km)")
        ax.set_title(f"R² = {r2:.3f}")
        st.pyplot(fig)

    with col2:
        st.write("### 📊 Residuals Distribution")
        residuals = y - y_pred
        fig, ax = plt.subplots(figsize=(5,4))
        sns.histplot(residuals, bins=30, kde=True, ax=ax, color="purple")
        ax.set_xlabel("Prediction Error (g/km)")
        ax.set_title("Residuals Distribution")
        st.pyplot(fig)

    st.write("---")
    st.write("### 🚘 Your Vehicle vs Dataset Average")
    comparison_df = pd.DataFrame({
        "Category": ["Your Vehicle", "Dataset Average"],
        "CO₂ emissions (g/km)": [prediction, avg_emission]
    })
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x="Category", y="CO₂ emissions (g/km)", data=comparison_df, palette="Set2", ax=ax)
    st.pyplot(fig)

    # Save Prediction
    try:
        current_user_resp = sb.auth.get_user()
        current_user = getattr(current_user_resp, "user", None)
        if current_user:
            features_payload = input_data.iloc[0].to_dict()
            payload = {
                "user_id": current_user.id,
                "features": features_payload,
                "predicted_co2": float(prediction)
            }
            sb.table("co2_predictions").insert(payload).execute()
    except Exception as e:
        st.warning(f"Could not save prediction to history: {e}")

    # User Prediction History
    st.write("---")
    st.subheader("🗂️ Your recent predictions")
    try:
        current_user_resp = sb.auth.get_user()
        current_user = getattr(current_user_resp, "user", None)
        if current_user:
            hist = (
                sb.table("co2_predictions")
                .select("*")
                .eq("user_id", current_user.id)  # Only fetch current user's history
                .order("created_at", desc=True)
                .limit(20)
                .execute()
            )

            if hist and getattr(hist, "data", None):
                rows = []
                for r in hist.data:
                    row = {"When": r["created_at"], "Predicted CO₂ (g/km)": r["predicted_co2"]}
                    if r.get("features"):
                        row.update(r["features"])
                    rows.append(row)
                df_hist = pd.DataFrame(rows)
                st.dataframe(df_hist, use_container_width=True)
            else:
                st.caption("No saved predictions yet.")
    except Exception as e:
        st.warning(f"Could not load history: {e}")

# Footer
st.write("---")
st.markdown("Built with ❤️ using **Python, Streamlit, and Scikit-learn**  \nProject by: *Sajivan & Team*")
