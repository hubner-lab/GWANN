import logging

class Logger:


    _instance = None

    def __new__(cls, name: str = 'my_logger', log_file: str = 'logs/app.log', level: int = logging.DEBUG):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.__initialize_logger(name, log_file, level)
        return cls._instance


    def __initialize_logger(self, name: str, log_file: str, level: int):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

 
        file_handler = logging.FileHandler(log_file, mode='a')
        console_handler = logging.StreamHandler()

        file_handler.setLevel(level)
        console_handler.setLevel(level)


        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)


        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)


    def debug(self, message: str):
        self.logger.debug(message)


    def info(self, message: str):
        self.logger.info(message)


    def error(self, message: str):
        self.logger.error(message)


if __name__ == "__main__":
    # Example usage:
    logger1 = Logger('my_logger', 'logs/app.log')
    logger1.debug('This is a debug message')
    logger2 = Logger('my_logger', 'logs/app.log')  # This will return the same instance as logger1
    logger2.info('This is an info message')
    logger1.error('This is an error message')

    # Check if both logger1 and logger2 are the same instance
    print(logger1 is logger2)  # Output: True
