import json
import os

from telegram.ext import CommandHandler, MessageHandler
from telegram.ext import filters
from telegram.ext import ApplicationBuilder

from models.config import Config
from handlers import *


class SynonymsSteganographyBot:

    def __init__(self):
        self._config = None

        os.environ['OPENAI_API_KEY'] = self.config.openai_api_key
        self.app = ApplicationBuilder().token(self.config.bot_token).build()

        handlers = [
            # general
            CommandHandler('start', start_handler),
            CommandHandler('help', help_handler),
            CommandHandler('encode', encode_handler),
            MessageHandler(filters.ATTACHMENT, decode_handler),

            # filter
            MessageHandler(filters.VIDEO, document_filter),
            MessageHandler(filters.Document, document_filter),
            MessageHandler(filters.PHOTO, photo_filter),
            MessageHandler(filters.TEXT, text_filter),
            MessageHandler(filters.VOICE, voice_filter)
        ]

        for handler in handlers:
            self.app.add_handler(handler)

    @property
    def config(self):
        if not self._config:
            if not os.path.exists('config.json'):
                raise FileNotFoundError('Config file not found!')

            with open('config.json') as cfg:
                self._config = Config(**json.load(cfg))

        return self._config

    def run(self):
        if self.app.updater:
            print("Bot successfully started.")
            self.app.run_polling()
        else:
            raise AttributeError('Bot was configured wrong')


if __name__ == '__main__':
    app = SynonymsSteganographyBot()
    app.run()
