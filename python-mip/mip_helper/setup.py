from setuptools import setup
from setuptools import config

conf_dict = config.read_configuration('setup.cfg')

setup(
    name='mip_helper',
    version=conf_dict['metadata']['version'],
    zip_safe=False,
    url='https://github.com/LREN-CHUV/python-base-docker-images',
    description='Read/Write inputs/outputs from/to the MIP',
    long_description=conf_dict['metadata']['long_description'],
    author='Mirco Nasuti',
    author_email='mirco.nasuti@chuv.ch',
    license='Apache 2.0',
    packages=['mip_helper'],
    extras_require={
        'test': ['unittest', 'nose'],
    },
    install_requires=['psycopg2-binary>=2.7.4',
                      'sqlalchemy>=1.2.5',
                      'pandas==0.22.0'],
    include_package_data=True,
    classifiers=(
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: Unix',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
    )
)
