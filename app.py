import os
import json
from io import StringIO, BytesIO
from uuid import uuid4
import zipfile

import streamlit as st

from utils.app import check_password
from steganography import helper, core
from steganography.secret_key import is_secret_key_valid
from utils.constants import Procedures
from models.config import Config
from models.report import ReportModel
from utils.database import MongoDB

st.title("Synonyms Steganography")

if not check_password():
    st.stop()  # Do not continue if check_password is not True.


_config = Config.from_json("config.json")
os.environ["OPENAI_API_KEY"] = _config.openai_api_key
database = MongoDB(**_config.mongodb)

choice = st.selectbox("Select your procedure", Procedures.values)
with st.form(key="main_form"):
    if choice == Procedures.ENCODING:
        message = st.text_input("Please, enter your message...")
        binary_message = helper.binarize_message(message)
        st.write(
            f"The binary length of your message: {len(binary_message)}."
        )
    else:
        uploaded_file = st.file_uploader("Choose encoded message file")
        if uploaded_file is not None:
            message_request_uuid = uploaded_file.name.split("_")[0]

            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            container = stringio.read()
            if not container:
                st.write("The container cannot be empty!")
                st.stop()
            else:
                st.write("OK")

        uploaded_file = st.file_uploader("Choose secret key file")
        if uploaded_file is not None:
            secret_key_request_uuid = uploaded_file.name.split("_")[0]

            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            secret_key = json.loads(stringio.read())
            if not secret_key or not is_secret_key_valid(secret_key):
                st.write("The secret_key is empty or invalid!")
                st.stop()
            elif message_request_uuid != secret_key_request_uuid:
                st.write("The secret_key and message have different UUID")
                st.stop()
            else:
                st.write("OK")

    encoding_submit = st.form_submit_button(label="Run")

if encoding_submit and choice == Procedures.ENCODING:
    st.write("Encoding, please wait...")
    request_uuid = str(uuid4())

    container, encoded_message, secret_key, time_report, usage_report = core.encode_message(
        binary_message,
        bits_per_word=_config.bits_per_word,
        additional_bits=_config.additional_bits,
        binarize=False,
    )

    text = "Here is your container message and secret key. Please download them.\n"
    text += "Time spent:\n"
    text += "\n".join([f"{helper.to_plain_text(k)}: {v}" for k, v in time_report.items()])
    st.write(text)

    report = ReportModel(
        uuid=request_uuid,
        message=message,
        encoded_message=encoded_message,
        encoding_time_report=time_report,
        encoding_usage_report=usage_report,
        secret_key=secret_key,
        container=container
    )
    # database.execute(**report.save())

    buf = BytesIO()
    with zipfile.ZipFile(buf, "x") as zip_file:
        zip_file.writestr(f"{request_uuid}_encoded_message.txt", encoded_message)
        zip_file.writestr(f"{request_uuid}_secret_key.json", json.dumps(secret_key))

    st.download_button(
        label="Download encoded message and secret key",
        data=buf.getvalue(),
        file_name=f"{request_uuid}.zip",
        mime="application/zip",
    )

elif encoding_submit and choice == Procedures.DECODING:
    st.write("Decoding, please wait...")
    decoded_message, spent_time = core.decode_message(container, secret_key)
    if decoded_message:
        st.write(f"Here is your decoded message: {decoded_message}")
        st.write(f"Spent time on decoding: {spent_time} seconds")

    # report = ReportModel(**database.fetch_one(**ReportModel.get(message_request_uuid)))
    # report.decoded_message = decoded_message
    # report.decoding_time = spent_time
    # database.execute(**report.update())

st.write("Reload the page to repeat the procedure")
