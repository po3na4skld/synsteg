import json
import os

from telegram.ext import ContextTypes
from telegram import InputMediaDocument


async def send_artifacts(
        encoded_message: str,
        secret_key: list[dict[str, dict[str, str]]],
        context: ContextTypes.DEFAULT_TYPE, chat_id: int
):
    with open(f'{chat_id}_secret_key.json', 'w') as f:
        json.dump(secret_key, f)

    with open(f'{chat_id}_encoded_message.txt', 'w') as f:
        f.write(encoded_message)

    await context.bot.send_media_group(
        chat_id,
        [InputMediaDocument(media=open(f'{chat_id}_secret_key.json')),
         InputMediaDocument(media=open(f'{chat_id}_encoded_message.txt'))]
    )

    os.remove(f'{chat_id}_secret_key.json')
    os.remove(f'{chat_id}_encoded_message.txt')


def to_plain_text(text: str) -> str:
    return text.replace('_', ' ').capitalize()
