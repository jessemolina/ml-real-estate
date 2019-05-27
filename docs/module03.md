# Module 03: Feature Engineering

## Domain Knowledge

Engineered features should start by using your own (or others') expertise about the domain.

**Objectives:**

* Isolate information that may be important
* Make use of boolean masks to develop new features

Let's build a feature for 2 beds & 2 baths, which is a popular home option for investors. 

```code
# create a boolean feature
df['two_and_two'] = ((df.beds == 2) & (df.baths == 2)).astype(int)

# determine the percentage of properties that are two_and_two
df.two_and_two.mean() * 100
```

Taking domain knowledge into consideration, we can account for the housing market recession that occurred between 2010 and 2013. To do so, we can develop a new feature that indicates whether the property transaction fell in between the mentioned time period. 

```code
# create features based on domain knowledge
df['during_recession'] = df.tx_year.between(left=2010, right=2013).astype(int)

# determine the percentage of transactions that occurred during the recession
df.during_recession.mean() * 100
```

## Interaction Features

Engineer features based on interaction between two or more features. 

**Objectives:**

* With the use of domain knowledge, build arithmetic features

Let's build a feature that determines the property age at the time of the transaction. 

```code
# Use arithmetic to construct new features 
df['property_age'] = df.tx_year - df.year_built

# Check for improper data values
df.property_age.describe()
```

From our summary statistics, we can see that our minimum property value is -8. Let's determine the number properties who also have a negative age. 

```code
# Determine the number of observations that have descrepancies
df.property_age.lt(0).sum()
```

Once again taking domain knowledge into consideration, we can assume that some properties were purchased before they were constructed. Since our client is only interested in existing homes, we'll remove these exceptions. 

```code
# Remove observations that are negative in property age
df = df[df.property_age >= 0]

# review the remaining observations
len(df)
```

We may find value in building a score system that ranks properties based on the number of schools nearby and their quality score. To do so, we can engineer a feature that is a product of both 'num_schools' and 'median_school'.

```code
# Create feature for school score
df['school_score'] = df.num_schools * df.median_school
df.school_score.median()
```

## Sparse Classes

Reduce the number of sparse classes for categorical features.

* Check for and consolidate similar classes
* Group sparse classes into a single "Other" class

When consolidating sparse classes, the general rule of thumb is that each class has at least 50 observations; this is subjective to each data set and is not an actual guideline. 

```code
# review the categorical features in use
df.dtypes[df.dtypes == 'object']
```

Let's start with the 'exterior_walls' series. 

```code
# get value counts for each exterior_walls class
df.exterior_walls.value_counts()
```

Alternatively, we can build a seaborn countplot to visualize the disparity amongst the classes.

```code
# use a countplot to compare distribution for each feature class
sns.countplot(y=df.exterior_walls.sort_values())
```

Let's start consolidating similar classes. 

```code
# consolidate similar wood classes
df.exterior_walls.replace(to_replace=['Wood Shingle', 'Wood Siding'], value='Wood', inplace=True) 
```

Next, we'll consolidate sparse classes into a single 'Other' class. 

```code
df.exteriror_walls.value_counts()
```

```code
# group sparse classes into "other"
other_exterior_walls = ['Stucco', 'Concrete Block', 'Masonry', 'Other', 'Asbestos shingle']
df.exterior_walls.replace(to_replace=other_exterior_walls, value='Other', inplace=True)
```

Let's look at the classes by count once again. 

```code
# unique roof class value counts
df.roof.value_counts()
```

```code
# build a visual seaborn countplot of the exterior walls
sns.countplot(y='exterior_walls', data=df)
```

We will repeat the same process for the 'roof' series. 

```code
# get value counts for roof
df.roof.value_counts()
```

```code
# consolidate into the 'Composition Shingle' class
df.roof.replace(['Composition, 'Wood Shake/ Shingles'], 'Composition Shingle', inplace=True)

# list of classes to be replaced by 'Other'
other_classes = ['Other', 'Gravel/Rock', 'Roll Composition', 'Slate', 'Asbestos', 'Metal', 'Built-up']

# consolidate into the 'Other' class
df.roof.replace(other_roofs, 'Other', inplace=True)
```

```code
# get value counts for roof
df.roof.value_counts()
```

```code
sns.countplot(y='exterior_walls', data=df)
```

## Dummy Variables

In the case of Scikit-Learn, it can't directly handle categorical features. Instead, indicator variables must be created for every categorical class. 

For example, our roof class has 5 classes. 

```code
# view unique classes for roof
df.roof.unique()
```

For any one observation, we will instead need to specify whether the observation matches any of the classes. In essence, we will need to create a feature for each class and assign an indicator value.  

```code
# convert categorical variables into indicator variables
df = pd.get_dummies(df. columns=['exterior_walls', 'roof', 'property_type']
```

Our roof series has now been replaced with five new features. 

```code
# display new dummy variables 
df.filter(like='roof').head(5)
```

## Remove Unused

Remove unused or redundant features from the dataset.

* Remove features that don't make sense to pass into the machine learning algorithms.
* Remove features that are redundant 

Since we created the 'property_age' feature, both the 'tx_year' and 'year_built' can be dropped. 

```code
df = df.drop(columns=['tx_year', 'year_built'])
```

Let's export our updated dataset before moving on. 

```code
df.to_csv('../dataset/analytical_base_table.csv', index=None)
```

## Next Module

[04. Model Training](module04.ipynb)
