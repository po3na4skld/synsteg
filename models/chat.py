from telegram import Update


class Chat:

    def __init__(self, update: Update):
        try:
            self.id = update.effective_chat.id
        except:
            self.id = update.callback_query.message.chat.id


class Document:

    def __init__(self, update):
        kwargs = update.message.document.__dict__

        self.file_id = kwargs.get('file_id')
        self.file_unique_id = kwargs.get('file_unique_id')
        self.file_name = kwargs.get('file_name')
        self.mime_type = kwargs.get('mime_type')
        self.file_size = kwargs.get('file_size')


class Message:

    def __init__(self, update):
        kwargs = update.message.__dict__

        self.id = kwargs.get('message_id', '')
        self.date = kwargs.get('date', '')
        self.text = kwargs.get('text', '')


class Context:

    def __init__(self, context):
        self.context = context

    @property
    def bot(self):
        return self.context.bot
