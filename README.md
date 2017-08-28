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

Inherit this Docker image and use the database_connector library.

1. Add `import database_connector` in your Python script;
2. Call `database_connector.fetch_data()` to get the input data;
3. Call `database_connector.save_results(pfa, error, shape)` to store the results.

Other functions are provided like :

* Use `database_connector.get_var()` to get the dependent variable code;
* Use `database_connector.get_covars()` to get the independent continuous variables codes;
* Use `database_connector.get_gvars()` to get the independent polynominal variables codes;
* Use `database_connector.var_type(var)` to get the type of the variable 'var';
* Use `get_parameter(p)` to get the parameter value of 'p'.

For more information, have a look at the library documentation.


## Data format

The `database_connector.fetch_data()` function returns:
```
{
  'columns': [<var1>, <var2>, ...], 
  'data': [(<value for var1 row1>, <value for var2 row1>, ...), 
          (<value for var1 row2>, <value for var2 row2>, ...), 
          ...]
}
```

Example:

```
{
  'columns': ['lefthippocampus', 'alzheimerbroadcategory'], 
  'data': [(Decimal('2.8197'), 'AD'), 
           (Decimal('2.9537'), 'CN'), 
           (Decimal('2.8504'), 'AD')]
}
```

The `database_connector.var_type(var)` function returns:
```
{'type': <type>, 'values': [<cat1>, <cat2>, ...]}
```

If the variable is a of type 'real', then the 'values' field contains an empty array.

Examples:
```
{'type': 'real', 'values': []}
```
or
```
{'type': 'polynominal', 'values': ['AD', 'CN', 'Other']}
```
