from models.base import BaseModel


class Config(BaseModel):

    fields = [
        'uuid',
        'created',
        'modified',
        'bot',
        'owner',
        'openai_api_key'
        'mongodb',
        'followers',
        'admins'
    ]

    collection_name = 'configs'

    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.bot_token = kwargs['bot_token']
        self.owner = kwargs['owner']
        self.openai_api_key = kwargs['openai_api_key']

        self.followers = kwargs.get('followers', [self.owner])
        self.admins = kwargs.get('admins', [self.owner])

    def is_owner(self, chat_id):
        return chat_id == self.owner
