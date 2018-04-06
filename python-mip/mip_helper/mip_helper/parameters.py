#!/usr/bin/env python3

import logging
import re
import os

# TODO: should return a dict
def fetch_parameters():
    """Get parameters from env variables."""
    return _get_parameters()

# TODO: remove params_list arg
def get_param(params_list, param_name, param_type, default_value):
    """Extract param and convert it into proper type."""
    for p in params_list:
        if p["name"] == param_name:
            try:
                return param_type(p["value"])
            except ValueError:
                logging.info('%s cannot be cast as %s' % (p['value'], str(param_type)))
    logging.info("Using default value of parameter %s: %s" % (param_name, default_value))
    return param_type(default_value)


# TODO: remove params_list arg
def get_boolean_param(params_list, param_name, default_value):
    """Extract boolean parameter from input['parameters'].
    :param params_list: input['parameters']
    :param param_name:
    :param default_value:
    """
    for p in params_list:
        if p["name"] == param_name:
            try:
                return p["value"].lower() in ("yes", "true", "t", "1")
            except ValueError:
                logging.warning("%s cannot be cast to boolean !")
    logging.info("Using default value: %s for %s" % (default_value, param_name))
    return default_value

    # TODO: create util fn in mip_helper
    # get_params
#    model_parameters = {x['name']: x['value']for x in io_helper._get_parameters()}
#    model_type = model_parameters.pop('type', 'linear_model')

# *********************************************************************************************************************
# Private functions
# *********************************************************************************************************************

def _get_parameters():
    param_prefix = "MODEL_PARAM_"
    research_pattern = param_prefix + ".*"
    parameters = []
    for env_var in os.environ:
        if re.fullmatch(research_pattern, env_var):
            parameters.append({'name': env_var.split(param_prefix)[1], 'value': os.environ[env_var]})
    return parameters
