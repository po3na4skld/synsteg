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
        """
        Convert the object's specified attributes to a dictionary.

        This method creates a dictionary representation of the object by extracting
        the values of specified attributes listed in the 'fields' parameter.

        Args:
            fields (list): A list of attribute names to include in the resulting dictionary.
            modified (bool, optional): If True, apply modifications to the object before extracting
                attribute values. Defaults to False.

        Returns:
            dict: A dictionary containing the specified attribute names as keys and their
                  corresponding values from the object.
        """
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
        """Returns last created model instance"""
        instance = database.fetch_one(**{
            'collection': cls.collection_name,
            'query': {},
            'sort': [('created', -1)]
        })
        if instance:
            return cls(**instance[0]) if isinstance(instance, list) else cls(**instance)
        else:
            return None

    def __repr__(self) -> str:
        """
        Return a string representation of the object suitable for debugging.

        This method generates a string representation of the object by collecting
        the values of its attributes specified in the 'fields' attribute.

        Returns:
            str: A string representation of the object for debugging purposes.
        """
        items = {name: getattr(self, name) for name in self.fields}
        return f"{items!r}"
