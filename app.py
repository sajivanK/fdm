# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# --- Supabase auth helper (requires auth.py from previous step) ---
from auth import supabase_client, get_user
sb = supabase_client()

# =========================
# Load trained final model (pipeline + regression)
# =========================
model = joblib.load("final_model.pkl")

# Load your dataset (for average comparisons)
df = pd.read_csv("cleaned_vehicle_dataset.csv")

# =========================
# App Layout & Auth UI
# =========================
st.set_page_config(page_title="CO₂ Emissions Predictor", page_icon="🚗", layout="wide")

# --- Authentication UI (safe rerun + clear inputs) ---
st.sidebar.title("Account")

def safe_rerun():
    """
    Try to call streamlit rerun. If not available in this runtime,
    swallow the error so the app doesn't show a false 'login failed' message.
    """
    try:
        st.experimental_rerun()
    except Exception:
        # experimental_rerun not available — do nothing (no error shown)
        return

# Sidebar tabs (keeps main UI clean)
login_tab, register_tab = st.sidebar.tabs(["🔐 Login", "🆕 Register"])

with login_tab:
    # Use explicit session_state keys so we can clear them later
    login_email = st.text_input("Email", key="login_email")
    login_pwd = st.text_input("Password", type="password", key="login_pwd")
    if st.button("Login", key="login_btn"):
        try:
            sb.auth.sign_in_with_password({"email": login_email, "password": login_pwd})
            # Clear the fields so they don't persist
            st.session_state["login_email"] = ""
            st.session_state["login_pwd"] = ""
            st.success("Logged in — reloading...")
            safe_rerun()
        except Exception as e:
            msg = str(e)
            if "Email not confirmed" in msg:
                st.error("Your email is not confirmed. Check your inbox (or spam).")
                st.caption("If you didn't receive it, open Supabase → Authentication → Users → Resend confirmation.")
            else:
                st.error(f"Login failed: {e}")

with register_tab:
    reg_email = st.text_input("Email (new)", key="reg_email")
    reg_pwd = st.text_input("Password (new)", type="password", key="reg_pwd")
    if st.button("Create account", key="register_btn"):
        try:
            sb.auth.sign_up({"email": reg_email, "password": reg_pwd})
            # clear register fields
            st.session_state["reg_email"] = ""
            st.session_state["reg_pwd"] = ""
            st.success("Account created. If email confirmation is ON, check your email.")
        except Exception as e:
            st.error(f"Signup failed: {e}")

# Show user info & logout
user = get_user(sb)
if user:
    st.sidebar.success(f"Signed in as {user.email}")
    if st.sidebar.button("Logout", key="logout_btn"):
        try:
            sb.auth.sign_out()
        except Exception:
            pass
        # Clear any auth UI fields as well
        for k in ("login_email", "login_pwd", "reg_email", "reg_pwd"):
            if k in st.session_state:
                st.session_state[k] = ""
        safe_rerun()
else:
    st.info("Please log in or register to use the CO₂ predictor.")
    st.stop()

# ---------- Main App (protected) ----------
st.title("🚗 Vehicle CO₂ Emissions Predictor")
st.markdown("Predict vehicle **CO₂ emissions (g/km)** based on specifications using a Multiple Linear Regression model.")

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🔧 Enter Vehicle Specifications")

engine_size = st.sidebar.number_input("Engine size (L)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)
cylinders = st.sidebar.number_input("Number of Cylinders", min_value=3, max_value=12, step=1, value=4)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Ethanol", "Natural Gas", "Premium Petrol"])
combined_l_100km = st.sidebar.number_input("Combined (L/100 km)", min_value=2.0, max_value=25.0, step=0.1, value=8.5)

# Predict button
if st.sidebar.button("Predict"):
    # Convert input to DataFrame (make sure column names match training pipeline)
    input_data = pd.DataFrame({
        "Engine size (L)": [engine_size],
        "Cylinders": [cylinders],
        "Fuel type": [fuel_type],
        "Combined (L/100 km)": [combined_l_100km]
    })

    # Predict directly (model already has preprocessing inside)
    prediction = model.predict(input_data)[0]

    # =========================
    # Results Section
    # =========================
    st.subheader("🔮 Predicted CO₂ Emissions")
    st.metric("CO₂ emissions (g/km)", f"{prediction:.2f}")

    # Dataset average comparison
    avg_emission = df["CO2 emissions (g/km)"].mean()

    # Model evaluation (on dataset, for display)
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

    # =========================
    # Graphs Section
    # =========================
    col1, col2 = st.columns(2)

    # 1. Actual vs Predicted Scatter Plot
    with col1:
        st.write("### 📈 Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(5,4))
        sns.scatterplot(x=y, y=y_pred, alpha=0.6, ax=ax)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        ax.set_xlabel("Actual CO₂ emissions (g/km)")
        ax.set_ylabel("Predicted CO₂ emissions (g/km)")
        ax.set_title(f"R² = {r2:.3f}")
        st.pyplot(fig)

    # 2. Residuals Plot
    with col2:
        st.write("### 📊 Residuals Distribution")
        residuals = y - y_pred
        fig, ax = plt.subplots(figsize=(5,4))
        sns.histplot(residuals, bins=30, kde=True, ax=ax, color="purple")
        ax.set_xlabel("Prediction Error (g/km)")
        ax.set_title("Residuals Distribution")
        st.pyplot(fig)

    # 3. User vs Dataset Average Comparison
    st.write("---")
    st.write("### 🚘 Your Vehicle vs Dataset Average")
    comparison_df = pd.DataFrame({
        "Category": ["Your Vehicle", "Dataset Average"],
        "CO₂ emissions (g/km)": [prediction, avg_emission]
    })
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x="Category", y="CO₂ emissions (g/km)", data=comparison_df, palette="Set2", ax=ax)
    st.pyplot(fig)

    # =========================
    # Save prediction to Supabase (per-user)
    # =========================
    try:
        # fetch latest user object (not cached)
        current_user_resp = sb.auth.get_user()
        current_user = None
        if hasattr(current_user_resp, "user"):
            current_user = current_user_resp.user
        elif isinstance(current_user_resp, dict) and "user" in current_user_resp:
            current_user = current_user_resp["user"]

        if current_user is not None:
            # prepare features payload: try DataFrame -> dict fallback to manual dict
            try:
                features_payload = input_data.iloc[0].to_dict()
            except Exception:
                # fallback if input_data is not present / different structure
                features_payload = {
                    "Engine size (L)": engine_size,
                    "Cylinders": cylinders,
                    "Fuel type": fuel_type,
                    "Combined (L/100 km)": combined_l_100km
                }

            payload = {
                "user_id": current_user.id,
                "features": features_payload,
                "predicted_co2": float(prediction)
            }
            sb.table("co2_predictions").insert(payload).execute()
    except Exception as e:
        # Non-fatal: app still works even if saving fails
        st.warning(f"Could not save prediction to history: {e}")

    # =========================
    # Show user's recent history (optional)
    # =========================
    st.write("---")
    st.subheader("🗂️ Your recent predictions")
    try:
        hist = sb.table("co2_predictions") \
                 .select("*") \
                 .order("created_at", desc=True) \
                 .limit(20).execute()

        if hist and getattr(hist, "data", None):
            rows = []
            for r in hist.data:
                row = {"When": r["created_at"], "Predicted CO₂ (g/km)": r["predicted_co2"]}
                if r.get("features"):
                    # merge feature keys
                    for k, v in r["features"].items():
                        row[k] = v
                rows.append(row)
            df_hist = pd.DataFrame(rows)
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.caption("No saved predictions yet.")
    except Exception as e:
        st.warning(f"Could not load history: {e}")

# =========================
# Footer
# =========================
st.write("---")
st.markdown("Built with ❤️ using **Python, Streamlit, and Scikit-learn**  \nProject by: *Sajivan & Team*")
