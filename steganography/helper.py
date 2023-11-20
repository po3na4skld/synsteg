from utils.constants import N_ASCII_BITS, PUNCTUATION, BRACKETS


def binarize_message(message: str) -> str:
    return ''.join([bin(x)[2:].zfill(N_ASCII_BITS) for x in list(message.encode('ascii'))])


def get_text_from_binary(binary: str) -> str:
    bigint = int(binary, base=2)
    return bigint.to_bytes((bigint.bit_length() + 7) // N_ASCII_BITS, 'big').decode()


def clean_container(container: str) -> str:
    return container.translate(str.maketrans('', '', PUNCTUATION))


def check_endswith_special(token: str) -> (str, str):
    special_tokens = [',', '.', '!', '?', '...']
    for i in special_tokens:
        if token.strip().endswith(i):
            return i, token.strip()[:-len(i)]
    return '', token


def has_duplicates(replacement_token) -> bool:
    tokens = list(replacement_token.values())[0]
    key = list(replacement_token.keys())[0]
    if len(tokens) != len(set(tokens.values())) or key in tokens.values():
        return True
    return False


def remove_brackets(container: str) -> str:
    return container.translate(str.maketrans('', '', BRACKETS))


def divide_chunks(container, chunk_size):
    cleaned_container = remove_brackets(clean_container(container)).split()
    for i in range(0, len(cleaned_container), chunk_size):
        yield cleaned_container[i:i + chunk_size]


def is_token_replacable(token, replacement_token):
    if token.istitle() and token.lower() in replacement_token:
        return True
    if token in replacement_token:
        return True
    return False


def check_capitalization(token, replacement_token):
    if token.istitle() and token not in replacement_token:
        if token.lower() in replacement_token:
            return token.lower(), token.istitle()
    return token, token.istitle()


def check_is_end_of_word(sequence, next_idx):
    if next_idx >= len(sequence):
        return True
    else:
        return sequence[next_idx] == ' '
