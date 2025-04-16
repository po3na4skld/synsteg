from random import sample, randint

from utils.constants import N_ASCII_BITS, PUNCTUATION, BRACKETS, SPECIAL_TOKENS


def binarize_message(message: str) -> str:
    """Binarize an ASCII-encoded message."""
    return ''.join([bin(x)[2:].zfill(N_ASCII_BITS) for x in list(message.encode('ascii'))])


def get_text_from_binary(binary: str) -> str:
    """Convert binary-encoded text to a human-readable string."""
    bigint = int(binary, base=2)
    return bigint.to_bytes((bigint.bit_length() + 7) // N_ASCII_BITS, 'big').decode()


def clean_container(container: str) -> str:
    """Clean container from punctuation tokens"""
    return container.translate(str.maketrans('', '', PUNCTUATION))


def check_endswith_special(token: str) -> (str, str):
    """Check if the token ends with one of special tokens"""
    for i in SPECIAL_TOKENS:
        if token.strip().endswith(i):
            return i, token.strip()[:-len(i)]
    return '', token


def has_duplicates(replacement_token: dict) -> bool:
    """Check if the replacement token has duplicated values"""
    tokens = list(replacement_token.values())[0]
    if len(tokens) != len(set(tokens.values())):
        return True
    return False


def remove_brackets(container: str) -> str:
    """Remove brackets from container string"""
    return container.translate(str.maketrans('', '', BRACKETS))


def divide_chunks(text: str | list, chunk_size: int) -> list[str]:
    """Split string into chunks with specified chunk_size"""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def is_token_replacable(token: str, replacement_token: dict) -> bool:
    """Check if a token is replaceable based on given replacement_token dictionary."""
    if token.istitle() and token.lower() in replacement_token:
        return True
    if token in replacement_token:
        return True
    return False


def check_capitalization(token: str, replacement_token: dict) -> tuple[str, bool]:
    """
    Check the capitalization of a token and provide a replacement if needed.

    This function examines the capitalization of the input token and determines
    whether it should be replaced with an alternative from the given replacement_token
    dictionary. The replacement is case-sensitive, but if the lowercase version of the
    token is found in the replacement_token dictionary, the replacement is made while
    preserving the original capitalization.
    """
    if token.istitle() and token not in replacement_token:
        if token.lower() in replacement_token:
            return token.lower(), token.istitle()
    return token, token.istitle()


def check_is_end_of_word(sequence: str, next_idx: int) -> bool:
    """Check if the given index marks the end of a word in the given sequence."""
    if next_idx >= len(sequence):
        return True
    else:
        return sequence[next_idx] == ' '


def to_plain_text(text: str) -> str:
    """Helper function for better debugging"""
    return text.replace('_', ' ').capitalize()


def generate_random_sequences(sequence_length: int, n_sequences: int) -> list[str]:
    if n_sequences > 2**sequence_length:
        sequence_length = n_sequences
    return [
        "".join([str(i) for i in map(int, f'{n:0>32b}')])[-sequence_length:]
        for n in sample(range(2**sequence_length + 1), n_sequences)
    ]


def get_random_message(n_bits: int) -> str:
    return "".join(str(randint(0, 1)) for _ in range(n_bits))
