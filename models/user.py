from models.base import BaseModel


class UserModel(BaseModel):

    fields = ['chat_id', 'username', 'first_name', 'last_name']
    show_fields = ['username', 'first_name', 'last_name']

    collection_name = 'users'

    def __init__(self, **kwargs):
        super(UserModel, self).__init__(**kwargs)

        self.chat_id = kwargs.get('chat_id', kwargs.get('id', None))
        self.username = kwargs.get('username', '')

        self.first_name = kwargs.get('first_name', '')
        self.last_name = kwargs.get('last_name', '')

    @staticmethod
    def get_or_create(update):
        user = UserModel(**{
            'chat_id': update.message.chat.id,
            'username': update.message.chat.username,
            'first_name': update.message.chat.first_name,
            'last_name': update.message.chat.last_name
        })
        return user
