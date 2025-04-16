from pymongo import MongoClient
from pymongo.server_api import ServerApi

from utils.constants import PAGE_SIZE
from utils.constants import DBMethods


class MongoDB:

    methods = DBMethods.values

    def __init__(self, **kwargs):
        self.user = kwargs.get('username', '')
        self.password = kwargs.get('password', '')
        self.host = kwargs.get('host', 'localhost')
        self.port = kwargs.get('port', 27017)
        self.name = kwargs.get('name', kwargs['name'])

        if not self.user and not self.password:
            self.client = MongoClient(f"mongodb://{self.host}:{self.port}")[self.name]
        else:
            self.client = MongoClient(f"mongodb+srv://{self.user}:{self.password}@{self.host}")[self.name]

    def fetch_all(self, collection, query, return_fields=(), sort=None, limit=PAGE_SIZE, page=0):
        """
        Fetch all records from database
        :param collection: Name of collection. 'required': True, 'type': str, 'example': articles
        :param query: Query 'required': True, 'type': str, 'example': {'uuid': 'uuid_example'}
        :param return_fields: List of fields to return from query. 'required': False,
        'type': List[Text], 'example': ['uuid', 'created']
        :param sort: a list of (key, direction) pairs specifying the sort order for this query
        'required': False, 'type': list, 'example': ['field_name', 1)] 1 == ASCENDING, -1 == DESCENDING
        :param limit: Limit size or page size. 'required': False, 'type': int, 'example': 20
        :param page: Page number, 0 - default. From which page return records. 'required': False,
        'type': int, 'example': 1
        :return: list of documents.
        """

        offset = page * limit

        if return_fields:
            result = self.client[collection].find(query, return_fields, sort=sort, limit=limit, skip=offset)
        else:
            result = self.client[collection].find(query, sort=sort, limit=limit, skip=offset)

        return [doc for doc in result]

    def fetch_one(self, collection, query, return_fields=(), sort=None):
        """
        Fetch one record from database
        :param collection: Name of collection. 'required': True, 'type': str, 'example': articles
        :param query: Query 'required': True, 'type': str, 'example': {'uuid': 'uuid_example'}
        :param return_fields: List of fields to return from query. 'required': False,
        'type': List[Text], 'example': ['uuid', 'created']
        :param sort: a list of (key, direction) pairs specifying the sort order for this query
        'required': False, 'type': list, 'example': ['field_name', 1)] 1 == ASCENDING, -1 == DESCENDING
        :return: list of documents.
        """

        if return_fields:
            result = self.client[collection].find_one(query, return_fields, sort=sort)
        else:
            result = self.client[collection].find_one(query, sort=sort)

        return dict(result) if result else None

    def execute(self, collection, method, query=None, docs=None):
        """
        Executes a special query
        :param collection: Name of collection. 'required': True, 'type': str, 'example': articles
        :param method: Execution method. 'required': True, 'type': str, 'example': insert
        :param query: Query. 'required': True, 'type': str, 'example': {'uuid': 'uuid_example'}
        :param docs: a list of documents to insert or update 'required': False,
        'type': list if many documents, dict if only one, 'example': [{'uuid': 'uuid_example}]

        :return: Doesn't returns anything.
        """
        assert method in self.methods, f"Invalid type argument. Must be in {self.methods}"

        if method == DBMethods.INSERT:
            assert docs is not None, 'docs should be specified when using insert or update methods'
            if isinstance(docs, list):
                self.client[collection].insert_many(docs)
            else:
                self.client[collection].insert_one(docs)

        elif method == DBMethods.UPDATE:
            assert docs is not None, 'docs should be specified when using insert or update methods'
            assert query is not None, 'query should be specified when using insert or update methods'
            self.client[collection].update_one(query, {'$set': docs})

        else:  # delete
            assert query is not None, 'query should be specified when using insert or update methods'
            self.client[collection].delete_many(query)
