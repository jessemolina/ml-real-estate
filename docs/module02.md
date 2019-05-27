# Module 02: Data Cleaning

Our second phase entails the data cleaning process. This phase is a critical one since cleaner data compensates for a lot more than any algorithm could. 

## Check for Duplicates

Let's start by removing any duplicates. 

**Objectives:** 

* Remove any duplicate observations
* Account for changes, if any 

We'll start by reviewing our data frame shape, call for the removal of duplicates, and determine if any were removed. 
    
```code
# Drop duplicates and check on the data frame's shape for differences
df.shape
```
```code
df = df.drop_duplicates()
df.shape
```

Based on the fact that our data frame shape stayed the same, we can determine that there were no duplicates. 

## Structural Errors
    
Next, we want to check for and account for any structural errors.

**Objectives:**

* Replace any variables that should be binary indicators
* Fix typos and capitalization issues for categorical features
* Check for and consolidate mislabeled classes

In our first phase, we determined that our basement feature could be modified to show whether a property had a basement or not. We'll change this feature to replace the missing values with zero. 

```code
# display unique values for given series
df.basement.unique()

# replace all NaN values with zero
df['basement'] = df.basement.fillna(0)
```

Next, let's check for and fix any typos or capitalization errors. We can also start consolidating features, if necessary. 

```code
# build count plot to view roof distribution 
sns.countplot(y='roof', data=df)
```

```code
# replace categorical data
df.roof.replace(to_replace='composition', value='Composition', inplace=True)

df.roof.replace(['shake-shingle', 'asphalt,shake-shingle'], 'Shake Shingle', inplace=True)
```

```code
# validate the changes made
df.roof.sort_values().unique()
```

Let's repeat the process for our exterior walls. 

```code
# build countplot to visualize exterior wall distribution
sns.countplot(y='exterior_walls', data=df)
```

```code
# consolidate values
df.exterior_walls.replace('Rock, Stone', 'Masonry', inplace=True)
```

## Unwanted Outliers

Let's remove any unwanted outliers that can negatively impact our regression model. The removal of these outliers need to be justified, as removing the wrong ones can also be an issue. 

**Objectives:**

* Check for measurements that are unlikely to be real data 
* Determine if any outliers belong to a different population data set
* Consider outliers that may need not pertain to the problem at hand

Let's build a box plot to quickly visualize our distribution and to check for outliers. 

```code
# build seaborn boxplot for tx_price
sns.boxplot(df.tx_price)
```
Alternatively, we can make use of a violin plot to visualize both summary statistics and probability distribution. 

```code
# build a seaborn violin plot of the tx_price
sns.violinplot(x=df.lot_size, data=df)
```

We'll check other features for potential outliers.

```code
# violin plot of beds
sns.violinplot(df.beds)
```
```code
# violin plot of sqft
sns.violinplot(df.sqft)
```

```code
# violin plot of lot_size
sns.violinplot(df.lot_size)
```
Looking at our violin plot of the lot size, we can see that there may be a potential outlier. We can further investigate this by reviewing the series top values. 

```code
# sort values to review for extreme boundry outliers
df.lot_size.sort_values(ascending=False)[:5]
```
We can see that the top lot_size value is significantly greater than any of the other four values. Since it will be unlikely that our client will invest in properties this large, we can remove it from the dataset. 

```code
# use a boolean mask to remove unwanted onservations
df = df[df.lot_size <= 500000]

# review the dataframe shape to confirm its removal
df.shape
```

## Missing Categorical Data

Rather than dropping or imputing null values, we explicitly state the values are missing.

```code
# check for null values by object dtype
df.select_dtypes(include='object').isnull().sum()
```

```code
# loop through object dtypes and replace null values with 'missing'
for series in df.select_dtype(include='object'):
    df[series] = df[series].fillna('Missing')

# verify that all null values have been replaced
df.select_dtypes(include='object').isnull().sum()
```

## Missing Numeric Data

As with categorical data, we need to be explicit if there any missing values. 

* Create new boolean feature indicating whether values are missing
* Set all null values to zero

Let's start off by checking for any missing values.

```code
# get sum of all missing values for numeric features
df.select_dtypes(exclude='object').isnull().sum()
```

There doesn't appear to be any missing numerical data, so we are finished with the data cleaning phase. Before moving on, let's export our updated dataset.

```code
df.to_csv('../dataset/cleaned_df.csv', index=None)
```

## Next Module
[03. Feature Engineering](module03.ipynb)
