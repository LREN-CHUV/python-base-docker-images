import logging
import sys
import pandas as pd

from . import io_helper
from .errors import UserError
from .shapes import Shapes

EXIT_ON_ERROR_PARAM = "exit_on_error"
DEFAULT_EXIT_ON_ERROR = True


# TODO: move it to io_helper and build the dataframe directly from sql result
# Rename to fetch_dataframe
def create_dataframe(variables):
    """Create dataframe from variables.
    :param vars: indep_vars or [dep_var] from `fetch_data`
    :return: dataframe with data from all variables
    """
    df = {}
    for var in variables:
        # categorical variable - we need to add all categories to make one-hot encoding work right
        if is_nominal(var):
            df[var['name']] = pd.Categorical(var['series'], categories=var['type']['enumeration'])
        else:
            # infer type automatically
            df[var['name']] = var['series']
    X = pd.DataFrame(df)
    return X


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
            io_helper.save_results('', str(e), Shapes.ERROR)
            exit_on_error()

    return wrapped


def exit_on_error():
    """Return exit code 1 if env `exit_on_error` is True. """
    parameters = io_helper.fetch_parameters()
    exit_on_error = io_helper.get_boolean_param(parameters, EXIT_ON_ERROR_PARAM, DEFAULT_EXIT_ON_ERROR)
    if exit_on_error:
        sys.exit(1)


def is_nominal(var):
    return var['type']['name'] in ['binominal', 'polynominal']


def is_integer(var):
    return var['type']['name'] in ['integer']
