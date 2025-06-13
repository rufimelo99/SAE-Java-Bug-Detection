import logging
import sys
from typing import Dict, List, Literal

import structlog

MODES = Literal["dev", "prod"]
LEVELS = Literal["debug", "info", "warning", "error"]


class Logger:
    """
    A custom logger class to manage logging in different modes (development and production)
    using the structlog library.

    This logger is designed to provide a flexible logging interface that can be configured
    based on the environment (dev or prod). It supports various log levels, allows for binding
    and unbinding of context, and formats log messages appropriately for the specified mode.

    Parameters
    ----------
    mode : Literal["dev", "prod"], optional
        The logging mode to use. Defaults to "dev". In "dev" mode, logs are more verbose and
        formatted for easy reading. In "prod" mode, logs are formatted for machine readability.

    min_level : Literal["debug", "info", "warning", "error"], optional
        The minimum log level for the logger. Defaults to "info". This level determines which
        log messages will be recorded. Messages below this level will be ignored.

    Attributes
    ----------
    logger : structlog.BoundLogger
        The structlog logger instance configured based on the specified mode and minimum log level.

    Methods
    -------
    bind(**kwargs)
        Binds the given key-value pairs to the logger context.

    unbind(keys: List[str])
        Unbinds the specified keys from the logger context.

    debug(message: str, **kwargs)
        Logs a debug-level message.

    info(message: str, **kwargs)
        Logs an info-level message.

    warning(message: str, **kwargs)
        Logs a warning-level message.

    error(message: str, **kwargs)
        Logs an error-level message.

    Notes
    -----
    - In development mode, the logger produces colored and formatted logs suitable for debugging.
    - In production mode, logs follow a structured format ideal for logging to files or log management systems.
    - The logging levels are mapped to their respective integer values using the standard logging library.

    Example
    -------
    logger = Logger(mode="dev", min_level="debug")
    logger.info("This is an info message")
    logger.bind(user_id=123)
    logger.error("This is an error message with user context")
    logger.unbind(["user_id"])
    """

    def __init__(self, mode: MODES = "dev", min_level: LEVELS = "info") -> None:
        self.mode = mode
        self.min_level = min_level

        if self.mode == "dev":
            self.__build_dev_logger(min_level)
        else:
            self._build_prod_logger(min_level)

        self.logger = structlog.get_logger()

    def bind(self, **kwargs) -> None:
        """
        Description:
            * Binds the given key-value pairs to the logger.

        Args:
            * kwargs: The key-value pairs to bind to the logger.
        """
        self.logger = self.logger.bind(**kwargs)

    def unbind(self, keys: List[str]) -> None:
        """
        Description:
            * Unbinds the given keys from the logger.

        Args:
            * keys: The list of keys to unbind from the logger.
        """
        self.logger = self.logger.unbind(*keys)

    def debug(self, message: str, **kwargs) -> None:
        """
        Description:
            * Logs a debug message. Debug messages are used for development purposes and are not
              shown in production. They should log information that is useful for debugging.

        Args:
            * message: The message to log.
            * kwargs: The key-value pairs to log.
        """
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """
        Description:
            * Logs an informative message. Informative messages are used to log information that is
              useful for understanding the state of the application or to mark an event.

        Args:
            * message: The message to log.
            * kwargs: The key-value pairs to log.
        """
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """
        Description:
            * Logs a warning message. Warning messages are used to log information that could be used
              to prevent an error from occurring. For instance, losing and reconnecting to a database,
              or a non-critical exception that is still meaningful.

        Args:
            * message: The message to log.
            * kwargs: The key-value pairs to log.
        """
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """
        Description:
            * Logs an error message. Error messages are used to log information about an error that
              occurred in the application. This kind of logs should be used to wake people up and
              trigger alerts. An example of an error message is a database connection error.

        Args:
            * message: The message to log.
            * kwargs: The key-value pairs to log.
        """
        self.logger.error(message, **kwargs)

    def __build_dev_logger(self, min_level: LEVELS) -> None:
        """
        Description:
            * Configures the logger for development mode. In development mode, the logger is more verbose
              and logs are colored and well formatted.

        Args:
            * min_level: The minimum log level to log.
        """
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.dev.ConsoleRenderer(
                    colors=True,
                ),
            ],
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(sys.stdout),
            wrapper_class=structlog.make_filtering_bound_logger(
                self.__literal_to_level(min_level)
            ),
            cache_logger_on_first_use=True,
        )

    def _build_prod_logger(self, min_level: LEVELS) -> None:
        """
        Description:
            * Configures the logger for production mode. In production mode, the logger is less verbose
              and logs follow the "logfmt" format.

        Args:
            * min_level: The minimum log level to log.
        """
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.EventRenamer("msg"),
                structlog.processors.TimeStamper(
                    fmt="%Y-%m-%d %H:%M:%S",
                ),
                structlog.processors.KeyValueRenderer(
                    sort_keys=True,
                    key_order=["level", "msg", "timestamp"],
                    drop_missing=False,
                ),
            ],
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(sys.stdout),
            wrapper_class=structlog.make_filtering_bound_logger(
                self.__literal_to_level(min_level)
            ),
            cache_logger_on_first_use=True,
        )

    @staticmethod
    def __literal_to_level(level: LEVELS) -> int:
        """
        Description:
            * Converts the given log level literal to the corresponding integer value. The logging library
              does not expose this method.

        Args:
            * level: The log level literal to convert.
        """
        return getattr(logging, level.upper())


logger = Logger()
