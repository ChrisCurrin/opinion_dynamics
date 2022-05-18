import logging


class LoggingContext:
    def __init__(self, level, logger="root"):
        self.logger = logging.getLogger(logger)
        self.logging_level = self.logger.getEffectiveLevel()
        self.logger.setLevel(level)

    def __enter__(self):
        return self.logger

    def __exit__(self, type, value, traceback):
        self.logger.setLevel(self.logging_level)