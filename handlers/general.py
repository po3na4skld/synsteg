import json
import os

from telegram import Update
from telegram.ext import ContextTypes

from models.chat import Chat
from models.user import UserModel
from steganography.core import encode_message, decode_message
from utils.chat import send_artifacts, to_plain_text
from steganography.secret_key import is_secret_key_valid


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = Chat(update)
    user = UserModel.get_or_create(update)

    text = f"Hello 👋, *{user.username}*!\n\n"
    text += "I'm your synonyms steganography bot.\n"
    text += 'Click /help for more information.'
    await context.bot.send_message(chat.id, text, parse_mode='markdown')


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = Chat(update)

    text = 'Here is the list of what I can do 📣:\n\n'
    text += '/encode {message} — Hide your message in text. I will return a filled container and secret key\n'
    text += '/decode {message} — Decode a message from a filled container and secret key, ' \
            'send your secret_key.json in a next message'

    await context.bot.send_message(chat.id, text)


async def encode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = Chat(update)
    encoded_message, secret_key, time_report = encode_message(' '.join(context.args))
    await send_artifacts(encoded_message, secret_key, context, chat.id)

    text = 'Here is your container message and secret key.\n'
    text += 'Time spent:\n'
    text += '\n'.join([f'{to_plain_text(k)}: {v}' for k, v in time_report.items()])
    await context.bot.send_message(chat.id, text)


async def decode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = Chat(update)
    if update.effective_message.caption.startswith('/decode'):
        message = ' '.join(update.effective_message.caption.split()[1:])
        new_file = await update.effective_message.effective_attachment.get_file()
        if not new_file.file_path.endswith('.json'):
            await context.bot.send_message('Secret key should be JSON file')
        else:
            await new_file.download_to_drive(f'{chat.id}_secret_key.json')
            with open(f'{chat.id}_secret_key.json') as f:
                secret_key = json.load(f)

            os.remove(f'{chat.id}_secret_key.json')
            if not is_secret_key_valid(secret_key):
                await context.bot.send_message(chat.id, "You've sent me an invalid secret_key.json")
            else:
                decoded_message = decode_message(message, secret_key)
                await context.bot.send_message(chat.id, f'Decrypted message: {decoded_message}')
    else:
        text = 'In order to decode your message use the next syntax:\n'
        text += '/decode {container_message} and attach your secret_key.json to this message'
        await context.bot.send_message(chat.id, text)
