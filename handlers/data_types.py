from telegram import Update
from telegram.ext import ContextTypes

from models.chat import Chat


async def photo_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = Chat(update)
    text = """I can't process photos yet."""
    return context.bot.send_message(chat.id, text)


async def document_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = Chat(update)
    text = """I can't process documents yet."""
    await context.bot.send_message(chat.id, text)


async def voice_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = Chat(update)
    text = """I can't process voice messages yet."""
    await context.bot.send_message(chat.id, text)


async def text_filter(update, context):
    chat = Chat(update)
    text = "I can't process text messages yet."
    await context.bot.send_message(chat.id, text)
