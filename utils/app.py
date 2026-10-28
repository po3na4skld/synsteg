import json
import os
from hashlib import sha512

import streamlit as st

PASSWORD_ENV = "SYNSTEG_PASSWORD_SHA512"  # hex SHA-512 of the app password; unset = no password gate
MONGODB_ENV = "SYNSTEG_MONGODB"  # JSON kwargs of utils.database.MongoDB; unset = reports are not stored


def check_password():
    """Returns `True` if no password is configured or the user entered the correct one."""
    password_hash = os.environ.get(PASSWORD_ENV)
    if not password_hash:
        return True

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        st.session_state["password_correct"] = (
            sha512(st.session_state["password"].encode("utf-8")).hexdigest() == password_hash.strip().lower())
        del st.session_state["password"]  # Don't store the password.

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Password", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


def get_database():
    """MongoDB from SYNSTEG_MONGODB, or None."""
    cfg = os.environ.get(MONGODB_ENV)
    if not cfg:
        return None
    from utils.database import MongoDB
    return MongoDB(**json.loads(cfg))
