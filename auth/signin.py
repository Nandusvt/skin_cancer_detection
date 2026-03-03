import streamlit as st
import pyrebase
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# FIREBASE CONFIG
# ─────────────────────────────────────────────
FIREBASE_CONFIG = {
    "apiKey":            os.getenv("FIREBASE_API_KEY"),
    "authDomain":        os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId":         os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket":     os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId":             os.getenv("FIREBASE_APP_ID"),
    "measurementId":     os.getenv("FIREBASE_MEASUREMENT_ID"),
    "databaseURL":       os.getenv("FIREBASE_DATABASE_URL", ""),
}

firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
auth = firebase.auth()

st.set_page_config(page_title="SkinAI — Login", layout="wide", page_icon="🔬")

if st.session_state.get("authenticated"):
    st.switch_page("app.py")
    st.stop()

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: radial-gradient(ellipse at 50% 0%, #0f2d5e 0%, #071428 55%, #020810 100%) !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
}
#MainMenu, header, footer { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 8vh 1rem 1rem !important; max-width: 100% !important; }

/* Card column */
[data-testid="column"]:nth-of-type(2) > div:first-child {
    background: rgba(10, 18, 38, 0.9) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 20px !important;
    padding: 44px 48px 40px !important;
    backdrop-filter: blur(12px);
    box-shadow: 0 24px 64px rgba(0,0,0,0.6);
}

/* Input fields */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 13px 16px !important;
}
.stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.35) !important; }
.stTextInput > div > div > input:focus {
    border-color: rgba(99,130,255,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,130,255,0.12) !important;
    background: rgba(255,255,255,0.09) !important;
}
.stTextInput > label { display: none !important; }

/* Primary button — blue gradient */
div[data-testid="stForm"] .stButton > button {
    background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 14px 20px !important;
    width: 100% !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s !important;
    margin-top: 8px !important;
}
div[data-testid="stForm"] .stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.4) !important;
}

/* Toggle / nav button — ghost */
.stButton > button {
    background: transparent !important;
    color: rgba(255,255,255,0.4) !important;
    border: none !important;
    font-size: 13px !important;
    padding: 4px 0 !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    color: white !important;
    background: transparent !important;
    transform: none !important;
    box-shadow: none !important;
}

[data-testid="stForm"] { border: none !important; padding: 0 !important; }
.stTextInput { margin-bottom: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "signin"

_, card_col, _ = st.columns([1, 1.05, 1])

with card_col:
    mode = st.session_state["auth_mode"]

    # Back arrow for sign-up
    if mode == "signup":
        if st.button("‹  Back to Sign in", key="back_arrow"):
            st.session_state["auth_mode"] = "signin"
            st.rerun()

    # Heading
    if mode == "signin":
        st.markdown(
            '<div style="color:white;font-size:28px;font-weight:900;margin:0 0 8px;letter-spacing:0px;text-align:center;">WELCOME BACK TO SkinAI!</div>'
            '<div style="color:rgba(255,255,255,0.4);font-size:13px;line-height:1.7;margin-bottom:26px;text-align:center;">'
            "Intelligent skin lesion analysis, powered by AI.<br>"
            "Sign in to begin secure dermoscopy evaluation."
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="color:white;font-size:28px;font-weight:900;margin:0 0 8px;">CREATE ACCOUNT</div>'
            '<div style="color:rgba(255,255,255,0.4);font-size:13px;line-height:1.7;margin-bottom:26px;">'
            "Fill in the details below to get started with SkinAI."
            '</div>',
            unsafe_allow_html=True
        )

    # SIGN IN
    if mode == "signin":
        with st.form("signin_form", clear_on_submit=False):
            email    = st.text_input("Email", placeholder="Email Address", autocomplete="email")
            password = st.text_input("Password", placeholder="Password",
                                      type="password", autocomplete="current-password")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if email and password:
                try:
                    user = auth.sign_in_with_email_and_password(email, password)
                    st.session_state["authenticated"] = True
                    st.session_state["user_email"]    = email
                    st.session_state["user_token"]    = user["idToken"]
                    st.success(f"Welcome back, {email}!")
                    st.switch_page("app.py")
                except Exception as e:
                    msg = str(e)
                    if "INVALID_PASSWORD" in msg or "INVALID_LOGIN_CREDENTIALS" in msg:
                        st.error("Invalid email or password.")
                    elif "EMAIL_NOT_FOUND" in msg:
                        st.error("No account found with this email.")
                    elif "TOO_MANY_ATTEMPTS" in msg:
                        st.error("Too many attempts. Try again later.")
                    else:
                        st.error("Sign in failed. Please try again.")
            else:
                st.warning("Please enter both email and password.")

        st.markdown(
            '<p style="text-align:center;color:rgba(255,255,255,0.38);font-size:13px;margin:18px 0 6px;">'
            "Don''t have an account? "
            
            '</p>',
            unsafe_allow_html=True
        )
        if st.button("Sign Up", key="go_signup", use_container_width=True):
            st.session_state["auth_mode"] = "signup"
            st.rerun()

        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;margin:18px 0 16px;">'
            '<div style="flex:1;height:1px;background:rgba(255,255,255,0.09);"></div>'
            '<span style="color:rgba(255,255,255,0.3);font-size:12px;white-space:nowrap;">Or continue with</span>'
            '<div style="flex:1;height:1px;background:rgba(255,255,255,0.09);"></div>'
            '</div>'
            '<div style="display:flex;justify-content:center;">'
            '<div style="width:50px;height:50px;border-radius:50%;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);display:flex;align-items:center;justify-content:center;">'
            '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 48 48"><path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.08 17.74 9.5 24 9.5z"/><path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/><path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/><path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-3.58-13.47-8.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/><path fill="none" d="M0 0h48v48H0z"/></svg>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # SIGN UP
    else:
        with st.form("signup_form", clear_on_submit=True):
            new_email  = st.text_input("Email", placeholder="Email Address", autocomplete="email")
            new_pass   = st.text_input("Password", placeholder="Password (min 6 characters)",
                                        type="password", autocomplete="new-password")
            new_pass2  = st.text_input("Confirm", placeholder="Confirm Password",
                                        type="password", autocomplete="new-password")
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            signup_submitted = st.form_submit_button("Create Account", use_container_width=True)

        if signup_submitted:
            if new_email and new_pass and new_pass2:
                if new_pass != new_pass2:
                    st.error("Passwords do not match.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        auth.create_user_with_email_and_password(new_email, new_pass)
                        st.success("Account created! Please sign in.")
                        st.session_state["auth_mode"] = "signin"
                        st.rerun()
                    except Exception as e:
                        msg = str(e)
                        if "EMAIL_EXISTS" in msg:
                            st.error("Email already registered.")
                        elif "WEAK_PASSWORD" in msg:
                            st.error("Password is too weak.")
                        else:
                            st.error("Sign up failed. Please try again.")
            else:
                st.warning("Please fill in all fields.")
