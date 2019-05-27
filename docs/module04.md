---
title: Module 05 -  Model Training
---

Let's start by importing the libraries. 

```code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

%matplotlib inline
```

Next, we'll import a few different algorithms that we will use against our dataset. 

```code
# Import specified linear algorithms
from sklearn.linear_model import ElasticNet, Ridge, Lasso

# Import specified ensemble algorithms 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
```

Finally, let's import the dataset and check its shape.

```code
df = pd.read_csv('../dataset/analytical_base_table.csv')

df.shape
```

## Split Dataset

Data should be thought of as a limited resources. Data can be split for training and for testing, but the same data can't be used for both.

**Objectives:**

Let's split the data set using a function from scikit-learn

```code
# function for splitting data
from sklearn.model_selection import train_test_split
```

Before completing the data split, we'll need to separate our target variable and our input features. 

```code
# seperate object for our target variable
y = df.tx_price

# seperate object for input features
X = df.drop('tx_price', axis=1)
```

Next, we'll split the dataset with 20% of our observations set aside for testing. 

```code
# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# verify length of each set
len(X_train), len(X_test), len(y_train), len(y_test)
```

# Model Pipeline

The majority of algorithms often require that the data is preprocessed. 

**Objectives:**

* Standardize data set
* Set up pipelines

Since our model will be using cross-validation, we'll need to set up a pipelines. 

```code
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

Next, we'll create a pipelines dictionary with all of our algorithms. 

```code
pipelines = {
    'lasso' : make_pipeline(StandardScaler(), Lasso(random_state=123)),
    'ridge' : make_pipeline(StandardScaler(), Ridge(random_state=123)),
    'enet' : make_pipeline(StandardScaler(), ElasticNet(random_state=123)),
    'rf' : make_pipeline(StandardScaler(), RandomForestRegressor(random_state=123)),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=123))
}
```

## Declare Hyperparameters to Tune

Unlike standard parameters that are learned attributes from the training data, hyperparameters (sometimes referred to as model parameters) are manually modified prior to training. 

**Objectives:**

* Create hyperparameter grids as for each algorithm 

```code
# lasso hyperparameters
lasso_hyperparameters = {
    'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5 10]
}   

# ridge hyperparameters
ridge_hyperparameters = {
    'ridge__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5 10]
}

# elastic net hyperparameters
enet_hyperparameters = {
    'elasticnet__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5 10],
    'elasticnet__l1_ration' : [0.1, 0.3, 0.5, 0.7, 0.9]
}

# random forest hyperparameters
rf_hyperparameters = {
    'randomforestregressor__n_estimators' : [100, 200],
    'randomforestregressor__max_featres' : ['auto', 'sqrt', 0.33]
}

# gradient boost hyperparameters
gb_hyperparameters = {
    'gradientboostingregressor__n_estimators': [100, 200],
    'gradientboostingregressor__learning_rate' : [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth' : [1, 3, 5]
}

```

Next, we'll create a dictionary for all of the hyperparameters. 

```code
hyperparameters = {
    'rf' : rf_hyperparameters, 
    'gb' : gb_hyperparameters,
    'lasso' : lasso_hyperparameters,
    'ridge' : ridge_hyperparameters,
    'enet' : enet_hyperparameters
}
```

## Cross Validation 


```code
from sklearn.model_selection import GridSearchCV
```

```code
fitted_models = {}

for name, pipeline in pipelines.items():
    # create cross-validation object 
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1)

    # fite model on X_train, y_train
    model.fit(X_train, y_train)

    # store model in dictionary
    fitted_models[name] = model

    # print message after model has been fitted
    print(name, 'has been fitted.)
```

Finally, we'll want to verify that our models have been fitted correctly

```code
from sklearn.exceptions import NotFittedError

for name, model in fitted_models.items():
    try:
        pred = model.predict(X_test)
        print(name, 'has been fitted.')
    except NotFittedError as e:
        print(repr(e))
```

# Evaluate Models

An initial way to evaluate our models is by looking at their cross-validated score on the the training set. 

```code
# display the average R^2 score for each model
for name, model in fitted_models.items():
    print(name, model.best_score_)
```

We will want to calculate the R^2 score on the test set, so let's import it. 

```code
from sklearn.metrics import r2_score
```

As an alternative to validate our models, we can also assess their performance based on their mean absolute error (MAE).

```code
from sklearn.metrics import mean_absolute_error
```

Let's tests our models against our test data. 

```code
for name, model in fitted_modes.items():
    pred = model.predict(X_test)
    print(name)
    print('________')
    print('R^2:', r2_score(y_test, pred))
    print('MAE:', mean_absolute_error(y_test, pred))
```

Let's assess our models by answering the following questions:

* Which model had the highest R^2 on the test set?
    __rf, Random Forest Regressor__

* Which model has the lowed mean absolute error?
    __rf, Random Forest Regressor__


