"""SynSteg encoder/decoder as of 14.12.2023 (git 6a87992 == 9b6f07e for core/helper), used for reports/json_reports.
Differences from the current code: has_duplicates also skips a slot whose synonyms contain the original word; the
container is split after remove_brackets; the decoder has no 'original word left in place' fallback."""
import math
from itertools import product

PUNCTUATION, BRACKETS, SPECIAL = ',.?!', '[](){}', [',', '.', '!', '?', '...']


def endswith_special(token):
    for i in SPECIAL:
        if token.strip().endswith(i):
            return i, token.strip()[:-len(i)]
    return '', token


def has_duplicates(slot):
    tokens, key = list(slot.values())[0], list(slot.keys())[0]
    return len(tokens) != len(set(tokens.values())) or key in tokens.values()


def replacable(token, slot):
    return (token.istitle() and token.lower() in slot) or token in slot


def capitalization(token, slot):
    if token.istitle() and token not in slot and token.lower() in slot:
        return token.lower(), True
    return token, token.istitle()


def fix_size(tokens, n):
    return {''.join(i): t for i, t in zip(product('01', repeat=n), list(tokens.values())[:2 ** n])}


def words(container):
    return container.translate(str.maketrans('', '', BRACKETS)).split()


def encode(container, key, bits):
    """Old core.encode_message walk for a fixed key; the key is shortened in place on the last block, as in 2023.
    Returns (stego, bits used)."""
    out, idx = [], 0
    for token, slot in zip(words(container), key):
        special, token = endswith_special(token)
        if replacable(token, slot) and idx < len(bits) and not has_duplicates(slot):
            token, cap = capitalization(token, slot)
            size = len(next(iter(slot[token])))
            part = bits[idx: idx + size]
            if len(part) != size:
                slot[token] = fix_size(slot[token], len(part))
            new = slot[token][part]
            out.append((new.capitalize() if cap else new) + special)
            idx += len(part)
        else:
            out.append(token + special)
    return ' '.join(out), idx


def decode(stego, key):
    """Old core.decode_message (clean_output=False)."""
    s = stego.translate(str.maketrans('', '', PUNCTUATION))
    bits, cur = '', 0
    for slot in key:
        base = list(slot.keys())[0]
        mapping = sorted(list(slot.values())[0].items(), key=lambda x: len(x[1]), reverse=True)
        found = False
        for code, tok in mapping:
            if has_duplicates(slot):
                cur += len(base) + 1; found = True; break
            end = cur + len(tok) >= len(s) or s[cur + len(tok)] == ' '
            if tok.lower() == s[cur:cur + len(tok)].lower() and end:
                bits += code; cur += len(tok) + 1; found = True; break
        if not found:
            break
    return bits
