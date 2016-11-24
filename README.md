# Python Base Docker Images

## What is it ?

This is a collection of Docker images for Python.

## Images

* python-base : Python 3 (includes Conda)
* python-mip : python-base + psycopg2 + custom library to access MIP databases

## Usage

### Build

Run: `./build.sh`

### Integrate a new algorithm

1. Add `import database_connector` in your Python script;
2. Call `database_connector.fetch_data()` to get the input data;
3. Call `database_connector.save_results(pfa, error, shape)` to store the results.

For more information, have a look at the library documentation.
