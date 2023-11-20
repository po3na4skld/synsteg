import logging
import sys


class LogWrapper:
    def __init__(self, logger):
        self.logger = logger

    def info(self, *args, sep=' '):
        self.logger.info(sep.join("{}".format(a) for a in args))

    def debug(self, *args, sep=' '):
        self.logger.debug(sep.join("{}".format(a) for a in args))

    def warning(self, *args, sep=' '):
        self.logger.warning(sep.join("{}".format(a) for a in args))

    def error(self, *args, sep=' '):
        self.logger.error(sep.join("{}".format(a) for a in args))

    def critical(self, *args, sep=' '):
        self.logger.critical(sep.join("{}".format(a) for a in args))

    def exception(self, *args, sep=' '):
        self.logger.exception(sep.join("{}".format(a) for a in args))

    def log(self, *args, sep=' '):
        self.logger.log(sep.join("{}".format(a) for a in args))


def get_logger(name):
    name = name.split('/')[-1]
    logging.basicConfig(
            stream=sys.stdout,
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.DEBUG
    )

    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('telegram').setLevel(logging.INFO)
    logging.getLogger('apscheduler').setLevel(logging.INFO)
    logging.getLogger('gql').setLevel(logging.WARNING)
    logging.getLogger('web3').setLevel(logging.WARNING)
    logging.getLogger('uniswap').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('backoff').setLevel(logging.ERROR)

    return LogWrapper(logging.getLogger(name))
