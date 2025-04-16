from argparse import ArgumentParser
from uuid import uuid4
import os

from steganography import core, helper
from models import ReportModel
from utils.logger import get_logger
from utils.constants import MAX_BITS_PER_WORD, MAX_ADDITIONAL_BITS_MULTIPLIER, CONTAINER_BUFFER

BASE_PATH = "artifacts"
LOGGER = get_logger(__name__)

parser = ArgumentParser()

parser.add_argument(
    "-container_path", required=True, help="Path to a .txt file with container message", type=str
)
parser.add_argument(
    "-output_path", required=False, help="Path to folder for output artifacts", type=str, default=BASE_PATH
)
parser.add_argument(
    "-message_length", required=False, choices=[128, 256, 512], help="Length of random secret message in bits", type=int
)
parser.add_argument(
    "-bits_per_word", required=True, type=int, help="Number of bits to encode per one word"
)
parser.add_argument(
    "-additional_bits", required=False, type=int, default=0, help="Number of additional bits to encode per one word"
)
parser.add_argument(
    "-config_path", required=False, default="config.json", type=str, help="Path to config with hparams"
)
parser.add_argument(
    "-openai_key", required=True, type=str, help="OpenAI API key"
)

if __name__ == '__main__':
    args = parser.parse_args()

    # Creating artifacts folder and loading openai api key
    os.makedirs(args.output_path, exist_ok=True)
    os.environ["OPENAI_API_KEY"] = args.openai_key

    # Ensuring bits per word is not exceeding the limit
    assert args.bits_per_word <= MAX_BITS_PER_WORD, f"bits_per_word too big, max allowed: {MAX_BITS_PER_WORD}"

    # Ensuring additional bits is not exceeding the limit
    msg = "Too many additional bits"
    assert args.additional_bits + args.bits_per_word <= args.bits_per_word * MAX_ADDITIONAL_BITS_MULTIPLIER, msg
    LOGGER.info(f"Working with: {args.bits_per_word} and {args.additional_bits} additional bits per word.")

    # Reading the container
    with open(args.container_path) as f:
        container = f.read().strip()

    # Checking that the container is big enough to fix full message length
    LOGGER.info(f"Processing {args.message_length} message length")
    min_words_required = args.message_length // (args.bits_per_word + args.additional_bits) * 1.5 * CONTAINER_BUFFER
    additional_words_needed = min_words_required - len(container.split())
    msg = f"Container to small to fit {args.message_length} bits, add {additional_words_needed} more words."
    assert min_words_required > 0, msg

    # Generating random secret message
    message = helper.get_random_message(args.message_length)
    request_uuid = str(uuid4())

    # Running encoding and decoding procedure
    LOGGER.info("Running encoding step")
    container, encoded_message, secret_key, time_report, usage_report = core.encode_message(
        message,
        bits_per_word=args.bits_per_word,
        additional_bits=args.additional_bits,
        binarize=False,
        container=container,
    )

    LOGGER.info("Running decoding step")
    decoded_message, spent_time = core.decode_message(
        encoded_message,
        secret_key,
        clean_output=False
    )

    # Saving report to artifacts folder in JSON format
    report = ReportModel(
        uuid=request_uuid,
        message=message,
        encoded_message=encoded_message,
        encoding_time_report=time_report,
        encoding_usage_report=usage_report,
        secret_key=secret_key,
        container=container,
        decoding_time=spent_time,
        decoded_message=decoded_message
    )

    report.to_json(BASE_PATH)

    LOGGER.info("All processes finished!")
