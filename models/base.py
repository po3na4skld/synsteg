from uuid import uuid4
from datetime import datetime


class BaseModel:

    fields = []
    collection_name = None

    def __init__(self, **kwargs):
        self.uuid = kwargs.get('uuid', str(uuid4()))
        self.created = kwargs.get('created', datetime.now())
        self.modified = kwargs.get('modified', datetime.now())

    def modify(self):
        self.modified = datetime.now()

    def to_dict(self, fields, modified=False):
        if modified:
            self.modify()

        return {key: getattr(self, key, None) for key in fields}

    def save(self):
        """Returns query to save or update model record in database"""
        return {
                'collection': self.collection_name,
                'docs': self.to_dict(self.fields),
                'method': 'insert'
            }

    def update(self):
        return {
            'collection': self.collection_name,
            'docs': self.to_dict(self.fields, modified=True),
            'method': 'update',
            'query': {'uuid': self.uuid}
        }

    def delete(self):
        """Returns query to delete model record from database"""
        return {
            'collection': self.collection_name,
            'query': {'uuid': self.uuid},
            'method': 'delete'
        }

    @classmethod
    def get(cls, uuid):
        """Returns instance by uuid"""
        return {'collection': cls.collection_name, 'query': {'uuid': uuid}}

    @classmethod
    def filter(cls, values):
        """Returns query to filter data by values"""
        assert isinstance(values, dict), 'values should be a dictionary type'
        return {'collection': cls.collection_name, 'query': values}

    @classmethod
    def get_last(cls, database):
        instance = database.fetch_one(**{
            'collection': cls.collection_name,
            'query': {},
            'sort': [('created', -1)]
        })
        if instance:
            return cls(**instance[0]) if isinstance(instance, list) else cls(**instance)
        else:
            return None

    def __repr__(self):
        items = {name: getattr(self, name) for name in self.fields}
        return f"{items!r}"
