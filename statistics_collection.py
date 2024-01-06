import json
import random
from uuid import uuid4
import os

from tqdm import tqdm
from glob import glob

from steganography import core
from models.report import ReportModel
from models.config import Config
from utils.logger import get_logger

LOGGER = get_logger(__name__)
BASE_PATH = 'json_reports'
MESSAGE_LENGTHS = [64, 128, 256]
N_ITERATIONS = 100


def get_random_message(n: int) -> str:
    return "".join(str(random.randint(0, 1)) for _ in range(n))


def get_number_of_iterations_for_message_length(message_len: int):
    iterations_left = 0
    for item in glob(f'{BASE_PATH}/*.json'):
        with open(item) as f:
            report_data = json.load(f)
            if len(report_data['message']) == message_len:
                iterations_left += 1
    return iterations_left


if __name__ == '__main__':
    _config = Config.from_json('config.json')
    os.environ['OPENAI_API_KEY'] = _config.openai_api_key
    LOGGER.info('Config setup successfully!')

    os.makedirs(BASE_PATH, exist_ok=True)
    LOGGER.info(f'Created {BASE_PATH} local database')

    for message_length in MESSAGE_LENGTHS:
        n_iterations = get_number_of_iterations_for_message_length(message_length)

        LOGGER.info(f'Processing {message_length} message length')
        for iteration in tqdm(range(N_ITERATIONS - n_iterations)):
            message = get_random_message(message_length)
            request_uuid = str(uuid4())

            container, encoded_message, secret_key, time_report = core.encode_message(message, binarize=False)
            decoded_message, spent_time = core.decode_message(encoded_message, secret_key, clean_output=False)

            report = ReportModel(
                uuid=request_uuid,
                message=message,
                encoded_message=encoded_message,
                encoding_time_report=time_report,
                secret_key=secret_key,
                container=container,
                decoding_time=spent_time,
                decoded_message=decoded_message
            )

            report.to_json(BASE_PATH)

    LOGGER.info('All processes finished!')
