#!/usr/bin/env python3

import logging

def get_boolean_param(params_list, param_name, default_value):
    for p in params_list:
        if p["name"] == param_name:
            try:
                return p["value"].lower() in ("yes", "true", "t", "1")
            except ValueError:
                logging.warning("%s cannot be cast to boolean !")
    logging.info("Using default value: %s for %s" % (default_value, param_name))
    return default_value

    # TODO: create util fn in mip_helper
#    model_parameters = {x['name']: x['value']for x in io_helper._get_parameters()}
#    model_type = model_parameters.pop('type', 'linear_model')
