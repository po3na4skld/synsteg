from models.base import BaseModel


class Config(BaseModel):

    fields = [
        'uuid',
        'created',
        'modified',
        'openai_api_key'
        'mongodb',
    ]

    collection_name = 'configs'

    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.openai_api_key = kwargs['openai_api_key']
        self.mongodb = kwargs['mongodb']
