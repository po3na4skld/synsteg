import time

from steganography import helper
from steganography.gpt import generate_container
from steganography.secret_key import generate_secret_key_mp, align_container_and_secret_key, fix_token_container_size
from utils.constants import SYNONYM_MAP


def encode_message(message: str) -> (str, SYNONYM_MAP, dict[str, float]):
    time_report = {}
    binary_message = helper.binarize_message(message)

    s = time.time()
    container = generate_container(len(binary_message))
    time_report['container_generation'] = round(time.time() - s, 2)

    s = time.time()
    container_splits = helper.divide_chunks(container, 100)
    secret_key = generate_secret_key_mp(container_splits)
    time_report['secret_key_generation'] = round(time.time() - s, 2)

    secret_key = align_container_and_secret_key(container, secret_key)

    encoded_message, current_idx = [], 0
    for token, replacement_token in zip(helper.remove_brackets(container).split(), secret_key):
        ends_with_special, token = helper.check_endswith_special(token)
        if helper.is_token_replacable(token, replacement_token) and current_idx < len(binary_message):  # Ingest token
            if helper.has_duplicates(replacement_token):
                encoded_message.append(token + ends_with_special)
                continue

            token, was_capital = helper.check_capitalization(token, replacement_token)
            token_container_size = len(list(replacement_token[token].keys())[0])
            message_part = binary_message[current_idx:current_idx + token_container_size]
            if len(message_part) != token_container_size:
                replacement_token[token] = fix_token_container_size(replacement_token[token], len(message_part))
            new_token = replacement_token[token][message_part]
            if was_capital:
                new_token = new_token.capitalize()

            encoded_message.append(new_token + ends_with_special)
            current_idx += token_container_size
        else:  # Skipping token
            encoded_message.append(token + ends_with_special)

    return ' '.join(encoded_message), secret_key, time_report


def decode_message(container: str, secret_key: SYNONYM_MAP) -> str:
    clean_encoded_message = helper.clean_container(container)
    binary_sequence, current_idx, counter = '', 0, 0
    for replacement_token in secret_key:
        base_token = list(replacement_token.keys())[0]
        mapping = sorted(
            [[k, v] for k, v in list(replacement_token.values())[0].items()],
            key=lambda x: len(x[1]),
            reverse=True
        )

        is_found = False
        for binary_data, token in mapping:
            span = clean_encoded_message[current_idx:current_idx + len(token)]
            is_end_of_word = helper.check_is_end_of_word(clean_encoded_message, current_idx + len(token))

            if helper.has_duplicates(replacement_token):
                current_idx += len(base_token) + 1
                is_found = True
                break

            if token.lower() == span.lower() and is_end_of_word:
                binary_sequence += binary_data
                current_idx += len(token) + 1
                is_found = True
                break

        counter += 1
        if not is_found:
            break

    return helper.get_text_from_binary(binary_sequence)
