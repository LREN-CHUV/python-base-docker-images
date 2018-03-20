# Copyright (C) 2017  LREN CHUV for Human Brain Project
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np


class Featurizer:
    """Class for preprocessing input data to numerical format required for scikit-learn. Can transform
    input dataframe and also generate PrettyPFA code.
    Inspired by Pipeline from scikit-learn. In the future we might use native scikit-learn Pipeline and
    convert it to PFA.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, data):
        return np.hstack([tf.transform(data) for tf in self.transforms])

    def generate_pretty_pfa(self):
        """Generate string for PrettyPFA that converts input to array of doubles."""
        transforms_pfa = ',\n    '.join([t.pfa() for t in self.transforms])
        return """
a.flatten(new(array(array(double)),
    {transforms}
))
        """.format(transforms=transforms_pfa).strip()


class Transform:
    """Transform implements methods `transform` and `pfa`.
    transform: X -> 2dim array
    pfa: generates PrettyPFA code
    """

    pass


class Standardize(Transform):

    def __init__(self, col, mu, sigma):
        self.col = col
        self.mu = mu
        self.sigma = sigma

    def transform(self, X):
        return ((X[self.col] - self.mu) / self.sigma)[:, np.newaxis]

    def pfa(self):
        return 'u.arr((input.{col} - {mu}) / {sigma})'.format(col=self.col, mu=self.mu, sigma=self.sigma)


class OneHotEncoding(Transform):

    def __init__(self, col, enumerations):
        self.col = col
        self.enumerations = enumerations

    def transform(self, X):
        Y = np.zeros((len(X), len(self.enumerations)))
        for i, val in enumerate(self.enumerations):
            Y[:, i] = X[self.col] == val
        return Y

    def pfa(self):
        categories = ','.join(['"{}"'.format(x) for x in self.enumerations])
        return 'u.C(input.{col}, new(array(string), {categories}))'.format(col=self.col, categories=categories)
