from multiprocessing import Pool
import math
from itertools import product
from typing import Any

import numpy as np

from steganography.gpt import get_openai_json_output
from steganography.helper import clean_container, remove_brackets, generate_random_sequences
from models.pool_arguments import PoolArguments, SecretKeyGenerationBody
from utils import prompts


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
                      "0" and "1" pointing to the base token.
    """

    if not len(synonyms):
        return {"0": base_token, "1": base_token}
    return {
        "".join(index): synonym
        for index, synonym in zip(product("01", repeat=int(math.log2(len(synonyms)))), synonyms)
    }


def binarize_synonyms_partially(
        synonyms: list[str], include_sequence: str, selected_synonym: str | None = None
) -> dict[str, str]:
    """
        Binarize a base token by assigning binary indices to synonyms so that include_sequence always will be used.

        This function takes a base token and a list of synonyms, then generates a dictionary
        where each synonym is associated with a unique binary index.
        For example if maximum bits per word by default is 4, with this function, we can use more bits in binarization
        with len(include_sequence) lengths the only requirement is that include sequence will be
        included 100% in that set.

        Args:
            synonyms (list[str]): A list of synonyms to be assigned binary indices.
            include_sequence (str): A binary sequence to include in binarization.
            selected_synonym (str | None): A synonym to map to include_sequence if specified

        Returns:
            dict[str, str]: A dictionary mapping binary indices to corresponding synonyms.
                          If the list of synonyms is empty, a default mapping is returned with
                          "0" and "1" pointing to the base token.
        """

    if not len(synonyms):
        raise ValueError("synonyms cannot be empty in this function")

    binary_sequences = generate_random_sequences(len(include_sequence), len(synonyms))
    if include_sequence not in binary_sequences:
        binary_sequences[np.random.randint(0, len(binary_sequences))] = include_sequence

    if selected_synonym:
        binarized_synonyms = {selected_synonym: include_sequence}
        binary_sequences.remove(include_sequence)
        for synonym, binary_sequence in zip([s for s in synonyms if s != selected_synonym], binary_sequences):
            binarized_synonyms[synonym] = binary_sequence
    else:
        binarized_synonyms = {synonym: binary_sequence for synonym, binary_sequence in zip(synonyms, binary_sequences)}

    return {binarized_synonyms[k]: k for k in sorted(list(binarized_synonyms), key=lambda _: np.random.random())}


def generate_synonyms(pool_arguments: PoolArguments) -> dict[str, list[str]]:
    """
    Generate synonyms for words related to the given context.

    Args:
        pool_arguments (PoolArguments): The context for which synonyms are to be generated.

    Returns:
        dict[str, list[str]]: A dictionary containing word categories as keys and lists of
        binarized synonyms as values. Each synonym list corresponds to a specific word category.
    """
    input_message = prompts.ALL_SYNONYMS_GENERATION_INPUT.format(context=pool_arguments.container_split)
    prompt = prompts.ALL_SYNONYMS_GENERATION_PROMPT.replace("N_SYNONYMS", str(2**pool_arguments.bits_per_word))
    synonyms, usage = get_openai_json_output(prompt, input_message, "words", 0.7)
    if synonyms and isinstance(synonyms[0], str):
        synonyms, usage = get_openai_json_output(prompt, input_message, "words", 0.7)

    return synonyms, usage


def generate_secret_key_mp(secret_key_generation_body: SecretKeyGenerationBody):
    """
    Generate a secret key using multiprocessing.

    This function takes a list of container chunks as input and utilizes the multiprocessing
    pool to concurrently generate secret key chunks using the `generate_synonyms` function.

    Args:
        secret_key_generation_body (SecretKeyGenerationBody): A list of container chunks,
            bits per word and additional bits used as input for secret key generation.

    Returns:
        list: A list of secret key chunks generated using multiprocessing.
    """

    secret_key, synonyms_chunks, is_filled = [], [], False
    usage_report = {
        "completion_tokens": 77,
        "prompt_tokens": 268,
        "total_tokens": 345,
    }

    with Pool() as pool:
        for (result, usage) in pool.imap(generate_synonyms, secret_key_generation_body.pool_arguments):
            synonyms_chunks += result
            usage_report["completion_tokens"] += usage["completion_tokens"]
            usage_report["prompt_tokens"] += usage["prompt_tokens"]
            usage_report["total_tokens"] += usage["total_tokens"]

    for idx, synonym in enumerate(synonyms_chunks):
        if is_filled:
            break

        for key, value in synonym.items():  # Always only one cycle
            if secret_key_generation_body.additional_bits:
                try:
                    binary_message_chunk = secret_key_generation_body.binary_message_chunks[idx]
                    partially_binarized = binarize_synonyms_partially(list(set(value)), binary_message_chunk)
                    secret_key.append({key: partially_binarized})
                except IndexError:
                    is_filled = True
            else:
                secret_key.append({key: binarize_synonyms(key, value)})

    return secret_key, usage_report


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
              {"0": token, "1": token} or the corresponding values from the secret key.
    """

    new_secret_key = []
    secret_key_idx = 0

    secret_key = clean_secret_key(secret_key)
    for idx, token in enumerate(clean_container(remove_brackets(container)).split()):
        try:
            secret_key_token = list(secret_key[secret_key_idx].keys())[0]
        except IndexError:
            new_secret_key.append({token: {"0": token, "1": token}})
            continue

        if token.lower() != secret_key_token.lower():
            new_secret_key.append({token: {"0": token, "1": token}})
        else:
            if len(list(secret_key[secret_key_idx].values())[0]) < 2:
                new_secret_key.append({token: {"0": secret_key_token, "1": secret_key_token}})
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
    return {"".join(i): t for i, t in zip(product("01", repeat=int(math.log2(new_tokens_size))), new_tokens)}


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
