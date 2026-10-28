import json
from io import BytesIO
from uuid import uuid4
import zipfile

import streamlit as st

from utils.app import check_password, get_database
from steganography import helper
from steganography.modes import Mode, PROFILES, encode, decode
from steganography.secret_key import is_secret_key_valid
from utils.constants import Procedures
from models.pool_arguments import LLMConfig
from models.report import ReportModel

OPENAI_MODELS = ["gpt-4o", "gpt-6-luna"]
OLLAMA_MODEL = "qwen3:30b-a3b-instruct-2507-q4_K_M"
LAYOUTS = ["b-bit table", "identity code (4.6)", "Hamming (7,3)", "variable bit width (4.5)"]

st.title("Synonyms Steganography")

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

with st.sidebar:
    st.header("Model")
    if st.radio("Provider", ["OpenAI", "Local (ollama)"], key="provider") == "OpenAI":
        llm = LLMConfig(provider="openai", model=st.selectbox("OpenAI model", OPENAI_MODELS),
                        api_key=st.text_input("OpenAI API key (empty = OPENAI_API_KEY)", type="password") or None)
    else:
        llm = LLMConfig(provider="ollama", model=st.text_input("Ollama model", OLLAMA_MODEL),
                        url=st.text_input("Ollama URL", "http://localhost:11434"))

    st.header("Mode")
    profile = st.selectbox("Profile (table 4.8)", ["custom"] + list(PROFILES), key="profile")
    p = Mode(**PROFILES.get(profile, {}))
    fixed = profile != "custom"
    b = st.slider("Bits per word b", 1, 5, p.bits_per_word)
    enhanced = st.radio("Mode", ["standard", "enhanced"], disabled=fixed) == "enhanced"
    additional = st.number_input("Additional bits", 1, 2 * b, 1) if enhanced and not fixed else 0
    st.caption("Improvements (subsection 4.3)")
    layout = st.radio("Slot layout", LAYOUTS, index=p.identity + 2 * p.hamming + 3 * p.variable_bits, disabled=fixed)
    ppl_choice = st.checkbox("Perplexity-guided choice (4.8)", p.ppl_choice, disabled=fixed)
    delta = st.number_input("δ", 0.0, 5.0, p.delta or 0.05, 0.01, disabled=fixed) \
        if st.checkbox("Perplexity window δ (4.7)", p.delta is not None, disabled=fixed) else None
    gamma = st.number_input("γ", 1.0, 50.0, p.gamma or 2.0, 0.5, disabled=fixed) \
        if st.checkbox("Perplexity target γ (4.7)", p.gamma is not None, disabled=fixed) else None
    key_selection = st.checkbox("Detector-guided key selection (4.3)", p.key_selection, disabled=fixed)
    mode = Mode(bits_per_word=b, additional_bits=additional, identity=layout == LAYOUTS[1], hamming=layout == LAYOUTS[2],
                variable_bits=layout == LAYOUTS[3], ppl_choice=ppl_choice, delta=delta, gamma=gamma,
                key_selection=key_selection)
    passphrase = st.text_input("Passphrase (identity, Hamming, key selection)", type="password", key="passphrase") or None

choice = st.selectbox("Select your procedure", Procedures.values)
with st.form(key="main_form"):
    if choice == Procedures.ENCODING:
        message = st.text_input("Please, enter your message...")
        container = st.text_area("Your own container (optional; empty = the model writes one)").strip() or None
        st.write(f"The binary length of your message: {len(helper.binarize_message(message))}.")
    else:
        stego_file = st.file_uploader("Choose encoded message file")
        key_file = st.file_uploader("Choose secret key file")
    submit = st.form_submit_button(label="Run")

if submit and choice == Procedures.ENCODING:
    try:
        with st.spinner("Encoding, please wait..."):
            r = encode(message, mode, llm, container=container, passphrase=passphrase)
    except (ValueError, ImportError, FileNotFoundError) as e:
        st.error(str(e))
        st.stop()
    request_uuid = str(uuid4())
    st.write(f"Bits embedded: {r['n_bits']} of the key's capacity {r['capacity_bits']} "
             f"({r['bits_per_word']:.2f} bits per container word); replaced words: {r['replaced_share']:.1%}")
    st.write("Time spent: " + ", ".join(f"{helper.to_plain_text(k)} {v} s" for k, v in r["time_report"].items()))
    if not r["recovered"]:
        st.warning("The stegotext does not decode back to the message with this key; encode again.")
    st.text_area("Stegotext", r["stegotext"])
    database = get_database()
    if database is not None:
        database.execute(**ReportModel(uuid=request_uuid, message=message, encoded_message=r["stegotext"],
                                       encoding_time_report=r["time_report"], encoding_usage_report=r["usage_report"],
                                       secret_key=r["keyfile"], container=r["container"]).save())

    buf = BytesIO()
    with zipfile.ZipFile(buf, "x") as zip_file:
        zip_file.writestr(f"{request_uuid}_encoded_message.txt", r["stegotext"])
        zip_file.writestr(f"{request_uuid}_secret_key.json", json.dumps(r["keyfile"]))
    st.download_button(label="Download encoded message and secret key", data=buf.getvalue(),
                       file_name=f"{request_uuid}.zip", mime="application/zip")

elif submit and choice == Procedures.DECODING:
    if stego_file is None or key_file is None:
        st.error("Upload both the encoded message and the secret key.")
        st.stop()
    stegotext = stego_file.getvalue().decode("utf-8")
    keyfile = json.loads(key_file.getvalue().decode("utf-8"))
    if not stegotext or not (isinstance(keyfile, dict) and keyfile.get("version") == 2 or is_secret_key_valid(keyfile)):
        st.error("The container is empty or the secret key is invalid!")
        st.stop()
    if stego_file.name.split("_")[0] != key_file.name.split("_")[0]:
        st.warning("The secret key and message have different UUIDs")
    try:  # the mode comes from the key file; the passphrase from the sidebar
        st.write(f"Here is your decoded message: {decode(stegotext, keyfile, passphrase)}")
    except ValueError as e:
        st.error(str(e))

st.write("Reload the page to repeat the procedure")
