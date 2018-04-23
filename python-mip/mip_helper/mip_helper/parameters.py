#!/usr/bin/env python3

import logging
import re
import os
import warnings


def fetch_parameters():
    """Get model parameters as dictionary from env variables."""
    param_prefix = "MODEL_PARAM_"
    research_pattern = param_prefix + ".*"
    parameters = {}
    for env_var in os.environ:
        if re.fullmatch(research_pattern, env_var):
            name = env_var.split(param_prefix)[1]
            value = os.environ[env_var]
            parameters[name] = value
    return parameters


def get_parameter(param_name, param_type, default_value):
    """Extract param and convert it into proper type."""
    params = fetch_parameters()
    if param_name in params:
        try:
            return param_type(params[param_name])
        except ValueError:
            logging.info('%s cannot be cast as %s' % (params[param_name], str(param_type)))
    logging.info("Using default value: %s for %s" % (default_value, param_name))
    return param_type(default_value)


def get_param(*args, **kwargs):
    warnings.warn('Deprecated, use get_parameters instead', DeprecationWarning)
    return get_parameter(*args, **kwargs)


def get_boolean_parameter(param_name, default_value):
    """Extract boolean parameter from input['parameters'].
    :param param_name:
    :param default_value:
    """
    return get_param(param_name, bool, default_value)


def get_boolean_param(*args, **kwargs):
    warnings.warn('Deprecated, use get_boolean_parameter instead', DeprecationWarning)
    return get_boolean_parameter(*args, **kwargs)
