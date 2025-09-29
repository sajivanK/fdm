# auth.py
import streamlit as st
from supabase import create_client, Client
from typing import Optional

@st.cache_resource
def supabase_client() -> Client:
    """
    Cached Supabase client. Uses keys in st.secrets:
      SUPABASE_URL
      SUPABASE_ANON_KEY
    """
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)

def get_user(sb: Client) -> Optional[dict]:
    """
    Returns the current authenticated user object or None.
    Use this each run (do NOT cache the returned user).
    """
    try:
        resp = sb.auth.get_user()
        # Depending on SDK version resp may be a dict or object.
        # Try both access patterns:
        user = None
        if hasattr(resp, "user"):
            user = resp.user
        elif isinstance(resp, dict) and "user" in resp:
            user = resp["user"]
        return user
    except Exception:
        return None
