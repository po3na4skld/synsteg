from multiprocessing import Pool
import math
from itertools import product
from typing import Any

from steganography.gpt import get_openai_json_output
from utils import prompts
from steganography.helper import clean_container, remove_brackets


def binarize_synonyms(base_token: str, synonyms: list[str]) -> dict[str, str]:
    """
    Binarize a base token by assigning binary indices to synonyms.

    This function takes a base token and a list of synonyms, then generates a dictionary
    where each synonym is associated with a unique binary index.

    Args:
        base_token (str): The base token to be binarized.
        synonyms (list[str]): A list of synonyms to be assigned binary indices.

    Returns:
        dict[str, str]: A dictionary mapping binary indices to corresponding synonyms.
                      If the list of synonyms is empty, a default mapping is returned with
                      '0' and '1' pointing to the base token.
    """

    if not len(synonyms):
        return {'0': base_token, '1': base_token}
    return {
        ''.join(index): synonym
        for index, synonym in zip(product('01', repeat=int(math.log2(len(synonyms)))), synonyms)
    }


def generate_synonyms(context: str) -> dict[str, list[str]]:
    """
    Generate synonyms for words related to the given context.

    Args:
        context (str): The context for which synonyms are to be generated.

    Returns:
        dict[str, list[str]]: A dictionary containing word categories as keys and lists of
        binarized synonyms as values. Each synonym list corresponds to a specific word category.
    """
    input_message = prompts.ALL_SYNONYMS_GENERATION_INPUT.format(context=context)
    synonyms = get_openai_json_output(prompts.ALL_SYNONYMS_GENERATION_PROMPT, input_message, "words")
    binarized_synonyms = []
    for synonym in synonyms:
        for key, value in synonym.items():
            synonym[key] = binarize_synonyms(key, value)
        binarized_synonyms.append(synonym)

    return binarized_synonyms


def generate_secret_key_mp(container_chunks):
    """
    Generate a secret key using multiprocessing.

    This function takes a list of container chunks as input and utilizes the multiprocessing
    pool to concurrently generate secret key chunks using the `generate_synonyms` function.

    Args:
        container_chunks (list): A list of container chunks used as input for secret key generation.

    Returns:
        list: A list of secret key chunks generated using multiprocessing.
    """

    secret_key_chunks = []
    with Pool() as pool:
        for result in pool.imap(generate_synonyms, container_chunks):
            secret_key_chunks += result
    return secret_key_chunks


def clean_secret_key(secret_key: list[dict]):
    """
    Cleans and sanitizes a list of dictionaries representing secret keys.

    Args:
        secret_key (list[dict]): A list of dictionaries where each dictionary represents a secret token
                                 and its corresponding mapping.

    Returns:
        list[dict]: A cleaned version of the secret key list, where each token and its mapping have
                    undergone sanitization using the clean_container function.
    """
    cleaned_secret_key = []
    for secret_token in secret_key:
        for token, mapping in secret_token.items():
            cleaned_secret_key.append({
                clean_container(token): {k: clean_container(v) for k, v in mapping.items()}
            })
    return cleaned_secret_key


def align_container_and_secret_key(container, secret_key):
    """
    Aligns a container and a secret key lengths, generating a new secret key list based on the alignment.
    Adding skipped words and removing duplicates while cleaning secret key.

    Args:
        container (str): The input container to align with the secret key.
        secret_key (str): The secret key to align with the container.

    Returns:
        list: A new list representing the aligned secret key, with each element being a dictionary.
              Each dictionary contains the token from the container as the key and a nested dictionary
              {'0': token, '1': token} or the corresponding values from the secret key.
    """

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
    """
    Adjusts the size of the token container to the specified new size.

    This function takes a dictionary of tokens and a new desired size for the token container.
    It truncates or pads the existing tokens to match the new size, creating a new dictionary
    with adjusted keys based on binary representations of indices.

    Args:
        tokens (dict[str, str]): A dictionary of tokens with original keys.
        new_token_container_size (int): The desired size for the new token container.

    Returns:
        dict[str, str]: A new dictionary of tokens with adjusted keys based on binary indices,
                       conforming to the specified container size.
    """
    new_tokens_size = 2 ** new_token_container_size
    new_tokens = list(tokens.values())[:new_tokens_size]
    return {''.join(i): t for i, t in zip(product('01', repeat=int(math.log2(new_tokens_size))), new_tokens)}


def is_secret_key_valid(secret_key: Any) -> bool:
    """
    Checks the validity of a secret key based on specified criteria.

    The function verifies that the provided secret_key adheres to a specific structure:
    - It should be a non-empty list of dictionaries.
    - Each dictionary should have string keys and dictionary values.
    - Each dictionary value should consist of string keys and string values.
    - The length of replacement_tokens within each dictionary should be consistent.

    Args:
        secret_key (Any): The secret key to be validated.

    Returns:
        bool: True if the secret key is valid; otherwise, False.
    """

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
