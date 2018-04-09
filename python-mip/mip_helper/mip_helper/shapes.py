#!/usr/bin/env python3

"""
Collection of mime types representing the valid shapes to use when saving the result of an algorithm
"""


class Shapes:
    ERROR = "text/plain+error"
    PFA = "application/pfa+json"
    PFA_YAML = "application/pfa+yaml"
    """
      Tabular data resource
      See: https://frictionlessdata.io/specs/tabular-data-resource/
    """
    TABULAR_DATA_RESOURCE = "application/vnd.dataresource+json"
    HTML = "text/html"
    SVG = "image/svg+xml"
    PNG = "image/png;base64"
    HIGHCHARTS = "application/vnd.highcharts+json"
    VISJS = "application/vnd.visjs+javascript"
    PLOTLY = "application/vnd.plotly.v1+json"
    VEGA = "application/vnd.vega+json"
    VEGALITE = "application/vnd.vegalite+json"
    """ Generic Json, for other types of visualisations """
    JSON = "application/json"
    TEXT = "text/plain"
