from argparse import ArgumentParser
from uuid import uuid4
import os

from steganography import helper
from steganography.modes import Mode, PROFILES, encode, decode
from models import ReportModel
from models.pool_arguments import LLMConfig
from utils.logger import get_logger
from utils.constants import MAX_BITS_PER_WORD, MAX_ADDITIONAL_BITS_MULTIPLIER

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
    "-message_length", required=False, choices=[128, 256, 512], default=128, help="Length of random secret message in bits", type=int
)
parser.add_argument(
    "-bits_per_word", required=False, type=int, default=None, help="Number of bits to encode per one word (default 5 or the profile's)"
)
parser.add_argument(
    "-additional_bits", required=False, type=int, default=0, help="Number of additional bits to encode per one word (enhanced mode)"
)
parser.add_argument(
    "-openai_key", required=False, type=str, default=None, help="OpenAI API key (default: OPENAI_API_KEY)"
)
parser.add_argument("-provider", choices=["openai", "ollama"], default="openai", help="Model provider")
parser.add_argument("-model", default=None, help="Model name (openai: gpt-4o, gpt-6-luna; ollama: qwen3:30b-a3b-instruct-2507-q4_K_M)")
parser.add_argument("-ollama_url", default="http://localhost:11434", help="Ollama server URL")
parser.add_argument("-profile", choices=list(PROFILES), default=None, help="Preset of table 4.8 (flags below override it)")
parser.add_argument("-identity", action="store_true", help="Identity code (4.6), needs a passphrase")
parser.add_argument("-hamming", action="store_true", help="Hamming (7,3) matrix embedding, needs a passphrase")
parser.add_argument("-variable_bits", action="store_true", help="Variable bit width (4.5)")
parser.add_argument("-ppl_choice", action="store_true", help="Perplexity-guided synonym choice (4.8)")
parser.add_argument("-delta", type=float, default=None, help="Perplexity window delta (4.7)")
parser.add_argument("-gamma", type=float, default=None, help="Perplexity target gamma (4.7)")
parser.add_argument("-key_selection", action="store_true", help="Detector-guided key selection (4.3), needs a passphrase")
parser.add_argument("-passphrase", default=os.environ.get("SYNSTEG_PASSPHRASE"), help="Shared secret (default: SYNSTEG_PASSPHRASE)")


def mode_from_args(args) -> Mode:
    m = dict(PROFILES.get(args.profile, {}))
    m.update({k: True for k in ("identity", "hamming", "variable_bits", "ppl_choice", "key_selection") if getattr(args, k)})
    m.update({k: getattr(args, k) for k in ("delta", "gamma") if getattr(args, k) is not None})
    if args.bits_per_word is not None:
        m["bits_per_word"] = args.bits_per_word
    m["additional_bits"] = args.additional_bits
    return Mode(**m).check()


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    mode = mode_from_args(args)
    llm = LLMConfig(provider=args.provider, model=args.model, url=args.ollama_url, api_key=args.openai_key)

    # Ensuring bits per word and additional bits are not exceeding the limit
    assert mode.bits_per_word <= MAX_BITS_PER_WORD, f"bits_per_word too big, max allowed: {MAX_BITS_PER_WORD}"
    msg = "Too many additional bits"
    assert mode.additional_bits + mode.bits_per_word <= mode.bits_per_word * MAX_ADDITIONAL_BITS_MULTIPLIER, msg
    LOGGER.info(f"Working with mode {mode} on {args.provider} {args.model or ''}")

    with open(args.container_path) as f:
        container = f.read().strip()

    # Generating random secret message
    message = helper.get_random_message(args.message_length)
    request_uuid = str(uuid4())

    LOGGER.info("Running encoding step")
    r = encode(message, mode, llm, container=container, passphrase=args.passphrase, binarize=False)

    LOGGER.info("Running decoding step")
    decoded_message = decode(r["stegotext"], r["keyfile"], args.passphrase, as_text=False)
    LOGGER.info(f"Decoded == message: {decoded_message == message}; replaced words {r['replaced_share']:.1%}")

    # Saving report to the output folder in JSON format (secret_key = the key file, version 2)
    ReportModel(
        uuid=request_uuid,
        message=message,
        encoded_message=r["stegotext"],
        encoding_time_report=r["time_report"],
        encoding_usage_report=r["usage_report"],
        secret_key=r["keyfile"],
        container=r["container"],
        decoded_message=decoded_message,
    ).to_json(args.output_path)

    LOGGER.info("All processes finished!")
