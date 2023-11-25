from multiprocessing import Pool
import math
from itertools import product
from typing import Any

from steganography.gpt import get_openai_json_output
from utils import prompts
from steganography.helper import clean_container, remove_brackets


def binarize_synonyms(base_token: str, synonyms: list[str]) -> dict[str, str]:
    if not len(synonyms):
        return {'0': base_token, '1': base_token}
    return {
        ''.join(index): synonym
        for index, synonym in zip(product('01', repeat=int(math.log2(len(synonyms)))), synonyms)
    }


def generate_synonyms(context: str) -> dict[str, list[str]]:
    input_message = prompts.ALL_SYNONYMS_GENERATION_INPUT.format(context=context)
    synonyms = get_openai_json_output(prompts.ALL_SYNONYMS_GENERATION_PROMPT, input_message, "words")
    binarized_synonyms = []
    for synonym in synonyms:
        for key, value in synonym.items():
            synonym[key] = binarize_synonyms(key, value)
        binarized_synonyms.append(synonym)

    return binarized_synonyms


def generate_secret_key_mp(container_chunks):
    secret_key_chunks = []
    with Pool() as pool:
        for result in pool.imap(generate_synonyms, container_chunks):
            secret_key_chunks += result
    return secret_key_chunks


def clean_secret_key(secret_key: list[dict]):
    cleaned_secret_key = []
    for secret_token in secret_key:
        for token, mapping in secret_token.items():
            cleaned_secret_key.append({
                clean_container(token): {k: clean_container(v) for k, v in mapping.items()}
            })
    return cleaned_secret_key


def align_container_and_secret_key(container, secret_key):
    new_secret_key = []
    secret_key_idx = 0

    secret_key = clean_secret_key(secret_key)
    for idx, token in enumerate(clean_container(remove_brackets(container)).split()):
        try:
            secret_key_token = list(secret_key[secret_key_idx].keys())[0]
        except IndexError:
            new_secret_key.append({token: {'0': token, '1': token}})
            continue

        if token.lower() != secret_key_token.lower():
            new_secret_key.append({token: {'0': token, '1': token}})
        else:
            if len(list(secret_key[secret_key_idx].values())[0]) < 2:
                new_secret_key.append({token: {'0': secret_key_token, '1': secret_key_token}})
            else:
                new_secret_key.append(secret_key[secret_key_idx])
            secret_key_idx += 1
    return new_secret_key


def fix_token_container_size(tokens: dict[str, str], new_token_container_size: int) -> dict[str, str]:
    new_tokens_size = 2 ** new_token_container_size
    new_tokens = list(tokens.values())[:new_tokens_size]
    return {''.join(i): t for i, t in zip(product('01', repeat=int(math.log2(new_tokens_size))), new_tokens)}


def is_secret_key_valid(secret_key: Any) -> bool:
    if isinstance(secret_key, list) and secret_key:
        if isinstance(secret_key[0], dict) and secret_key[0]:
            for replacement_map in secret_key:
                for key_token, replacement_tokens in replacement_map.items():
                    if not isinstance(replacement_tokens, dict):
                        return False

                    if not isinstance(key_token, str) or not replacement_tokens:
                        return False

                    if len(set([len(str(k)) for k in replacement_tokens])) != 1:
                        return False

                    for k, v in replacement_tokens.items():
                        if not isinstance(k, str) or not k.isnumeric() or not isinstance(v, str):
                            return False
            return True
    return False
