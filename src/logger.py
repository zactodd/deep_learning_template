from patterns import Singleton
import utils
import logging


LOGGER_FILE_FORMAT = '%Y_%b_%d___%H_%M_%S'


class Logger(metaclass=Singleton):
    def __init__(self):
        self.logger = None
        self.start()

    def start(self):
        filename = f"{utils.DEFAULT_DATA_PATH}/log_{utils.time_now_str(LOGGER_FILE_FORMAT).lower()}.log"

        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(fmt)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.logger.propagate = False


logger = Logger().logger


def close_logger():
    global logger
    for handler in logger.handlers:
        handler.close()


def open_logger():
    global logger
    Logger().start()
    logger = Logger().logger
