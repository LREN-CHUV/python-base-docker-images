#!/usr/bin/env python3

import logging
import traceback
import sys
from raven import Client

from .errors import UserError
from .parameters import get_boolean_parameter

EXIT_ON_ERROR_PARAM = "exit_on_error"
DEFAULT_EXIT_ON_ERROR = True

# NOTE: set env SENTRY_DSN to send errors to sentry
sentry_client = Client()


def remove_nulls(X, errors='raise'):
    """Remove null values from dataframe.
    :param X: dataframe with input data
    :param errors: if `raise` then raise UserError when all values are null, silently pass for `ignore`
    """
    X = X.dropna()

    if errors == 'raise' and len(X) == 0:
        raise UserError('No data left after removing NULL values, cannot fit model.')
    return X


def catch_user_error(func):
    """Catch UserError and save its message as job result."""

    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except UserError as e:
            logging.error(e)
            from . import io_helper
            io_helper.save_error(str(e))
            exit_on_error()
        except Exception as e:
            logging.exception(e)
            from . import io_helper
            io_helper.save_error(traceback.format_exc())
            sentry_client.captureException()
            raise e

    return wrapped


def exit_on_error():
    """Return exit code 1 if env `exit_on_error` is True. """
    exit_on_error = get_boolean_parameter(EXIT_ON_ERROR_PARAM, DEFAULT_EXIT_ON_ERROR)
    if exit_on_error:
        sys.exit(1)


def is_nominal(var):
    return var['type']['name'] in ['binominal', 'polynominal']


def is_integer(var):
    return var['type']['name'] in ['integer']
