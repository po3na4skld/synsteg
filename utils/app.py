from hashlib import sha512

import streamlit as st

PASSWORD_HASH = """
49c958022a2c126e7d0f624fb0f9e048b7a277c8275aed85de10719cdf25c7c8f
2da117feb40463a46bd976fa2b7260b36eb39518258319d3b753b0534016b94""".replace('\n', '')


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        password_hash = sha512(st.session_state["password"].encode('utf-8')).hexdigest()
        if password_hash == PASSWORD_HASH:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False
