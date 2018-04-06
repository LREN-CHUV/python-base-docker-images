import logging
import sys

from . import io_helper
from .utils import get_boolean_param
from .errors import UserError
from .shapes import Shapes

EXIT_ON_ERROR_PARAM = "exit_on_error"
DEFAULT_EXIT_ON_ERROR = True


def catch_user_error(func):
    """Catch UserError and save its message as job result."""

    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except UserError as e:
            logging.error(e)
            io_helper.save_results('', str(e), Shapes.ERROR)
            exit_on_error()

    return func


def exit_on_error():
    """Return exit code 1 if env `exit_on_error` is True. """
    parameters = io_helper.fetch_parameters()
    exit_on_error = get_boolean_param(parameters, EXIT_ON_ERROR_PARAM, DEFAULT_EXIT_ON_ERROR)
    if exit_on_error:
        sys.exit(1)


def is_nominal(var):
    return var['type']['name'] in ['binominal', 'polynominal']


def is_integer(var):
    return var['type']['name'] in ['integer']
