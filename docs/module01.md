---
title: Module 01 - Exploratory Analysis
---

Let's begin our project by first getting to know the dataset. Our initial analysis will allow us to start planning for our next steps in data cleaning and feature engineering. This phase should be quick, but thorough enough for us to gain a basic intuition for addressing the problem at hand. 

## Basic Information 

**Objectives:**

* Import data as a pandas data frame.
* Review data shape (observations, features). 
* Review the data types to determine categorical versus numerical features. 
* Reference the data dictionary and verify features weren't imported as incorrect data type. 
* Perform base analysis of dataset and get a qualitative feel. 

Let's start by importing our libraries and make our initial configurations. 

```code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
        
# change pandas option to view additional data frame columns
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', None)

# display plots in the notebook
%matplotlib inline
```
Next, let's load our data set.

```code
# load the real estate data from the dataset folder
df = pd.read_csv('../dataset/real_estate_data.csv')
```

Now that our data set has been loaded, let's get an initial understanding on what we're working with. 

```code
# the dataframe shape tells us the number of observations and features available
df.shape
```

```code
# display the columns and sort them by index name
df.dtypes.sort_index()
```

```code
# determine which features are categorical
df.dtypes[df.dtypes == 'object']
```

```code
# display the first five observations
df.head(5)
```

```code
# display the last five observations
df.tail()
```

# Numeric Distribution
 
After completing our basic observations, let's review our numerical data.

**Objectives:**

* Check for unexpected distributions (e.g. max value higher than normal)
* Insepct for numbers outside of their boundaries (e.g. +100%)
* Look out for any sparse data and measurement errors

      
```code
# plot histogram 
df.hist(figsize=(14, 14), xrot=-45)

# clear the text "residue"
plt.show()
```
While building a visual provides a quick interpretation of the data, it lacks the detail necessary for a more in-depth analysis.

```code
# display summary statistics, such as mean, std, and quartiles
df.describe()
```

```code
# we can specify summary statistics for a particular observation
df.basement.describe()
```

At a quick glance, our numerical data appears to make sense. At this point, we can consider features that could potentially be replaced by booleans, such as the basement feature.

## Categorical Distribution
    
Next, let's review the categorical data.

**Objectives:**

* Observe class frequency
* Account for sparse data that could be combined or reassigned


```code
# filter our observations by object type to provide descriptions
df.describe(include=['object'])
```
From our object type descriptions, we see that some features contain multiple unique values. We can build a seaborn countplot visual to better understand their distribution. 

```code
# display a barplot with the count for y variables
sns.countplot(y='exterior_walls', data=df)
```

```code
# barplot with count for each object index
for index in df.dtypes[df.dtypes == 'object'].index:
    sns.countplot(y=index, data=df)
```
At this point, we should have started to think of features we can consider consolidating. 

## Segmentations

Review segmentation to observe the relationship between categorical and numeric features. 

**Objective:**

* Build a boxplot to segment the target variable (tx_price) by key categorical features. 


```code
# use boxplot for a visual interpretation
sns.boxplot(y='property_type', x='tx_price', data=df)
```

```code
# segment by property_type and get means for each class
df.groupby('property_type').mean()
```

```code
# segment by property_type and get both mean and std for each class
df.groupby('property_type').agg(['mean', 'std'])
```
When comparing our features, we should consider the following questions:

* On average, which type of property is larger?

* Which type of property has larger lots?

* Which property is in areas of more nightlife/restaurants/grocery stores?

* Are there any relationships that make intuitive sense?

## Correlation

Correlate relationships between numeric features against the target variable. 

**Objectives:**

* Search for strong correlations for target variable

To start, we can assess our correlations against our target value. 

```code
# Calculate correlation between numeric features
correlations = df.corr()
correlations.tx_price.sort_values(ascending=False)
```

Knowing that we have different property types, we can build correlations for each types for further analysis. 

```code
apt_correlation = df[df.property_type != 'Single-Family'].corr()
apt_correlation.tx_price.sort_values(ascending=False)
```

```code
single_correlation = df[df.property_type == 'Single-Family'].corr()
single_correlation.tx_price.sort_values(ascending=False)
```

We can create a Seaborn heatmap to better visualize the correlations. 

```code
# create pyplot figure
plt.figure(figsize=(10, 8))

# generate mask to create triangle figure
mask = np.zeros_like(correlations, dtype=np.bool) # 2d ndarry bool values, same shape as correlations

mask[np.triu_indices_from(mask)] = True # set upper triangle indices to True

# plot heatmap as triangle
sns.heatmap(correlations * 100, annot=True, fmt='.0f', mask=mask, cmap='RdBu')
```
## Next Module

[02. Data Cleaning](module02.ipynb)
