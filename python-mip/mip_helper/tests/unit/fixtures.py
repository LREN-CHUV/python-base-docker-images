import pandas as pd
import numpy as np


def metadata():
    # see https://github.com/HBPMedical/woken/blob/master/src/test/resources/metadata/mip_cde_variables.json for
    # all definitions
    return {
        'lefthippocampus': {
            'description': 'Left hippocampus.',
            'methodology': 'mip-cde',
            'label': 'Left hippocampus',
            'code': 'lefthippocampus',
            'units': 'mm3',
            'length': 3,
            'type': 'real',
            'mean': 3.0,
            'std': 0.35,
            'minValue': 0.,
            'maxValue': 10.,
        },
        'minimentalstate': {
            'code': 'minimentalstate',
            'type': 'real',
            'mean': 24.0,
            'std': 5.0,
            'minValue': 0.,
            'maxValue': 50.,
        },
        'opticchiasm': {
            'code': 'opticchiasm',
            'type': 'real',
            'mean': 0.08,
            'std': 0.009,
            'minValue': 0.,
            'maxValue': 1.,
        },
        'subjectage': {
            'code': 'subjectage',
            'type': 'real',
            'mean': 71.0,
            'std': 8.0,
            'minValue': 0.,
            'maxValue': 100.,
        },
        'agegroup': {
            "code": "agegroup",
            "description": "Age Group",
            "enumerations": [
                {
                    "code": "-50y",
                    "label": "-50y"
                }, {
                    "code": "50-59y",
                    "label": "50-59y"
                }, {
                    "code": "60-69y",
                    "label": "60-69y"
                }, {
                    "code": "70-79y",
                    "label": "70-79y"
                }, {
                    "code": "+80y",
                    "label": "+80y"
                }
            ],
            "label": "Age Group",
            "methodology": "mip-cde",
            "type": "polynominal"
        },
    }


def data():
    return pd.DataFrame(
        {
            'lefthippocampus': [np.nan, 3.4613, 3.3827, 3.1983, 2.6429],
            'minimentalstate': [np.nan, 25.0, 22.0, np.nan, 21.0],
            'opticchiasm': [np.nan, 0.08029, 0.08488, 0.05304, 0.06139],
            'subjectage': [np.nan, 63.0, 67.0, np.nan, 71.0],
            'agegroup': ["70-79y", "70-79y", "70-79y", "70-79y", "+80y"]
        }
    )


def var():
    return 'lefthippocampus'


def covars():
    return ['minimentalstate', 'opticchiasm', 'subjectage', 'agegroup']
