import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to convert ISO 8601 duration to minutes
def convert_duration_to_minutes(duration_str):
    if pd.isna(duration_str):
        return None  # Return None if value is NaN
    match = re.match(r'P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?', duration_str)
    days, hours, minutes = match.groups(default='0')
    return int(days) * 1440 + int(hours) * 60 + int(minutes)

# Load your dataset
data = pd.read_csv('recipes_trunc.csv')

# Convert time columns from ISO 8601 format to minutes
time_columns = ['CookTime', 'PrepTime', 'TotalTime']
for col in time_columns:
    data[col] = data[col].apply(convert_duration_to_minutes)

# Columns that are not necessary for model training (change as needed)
columns_to_drop = ['AuthorId', 'AuthorName', 'DatePublished', 'Images', 'ReviewCount']
data.drop(columns=columns_to_drop, inplace=True)

# Define numerical and categorical columns for further processing
numerical_columns = ['CookTime', 'PrepTime', 'TotalTime', 'AggregatedRating', 
                     'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                     'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent']
categorical_columns = ['RecipeCategory', 'Keywords']

# Make sure 'Calories' is retained and not processed in ColumnTransformer
necessary_columns = ['RecipeId', 'Name', 'Description', 'RecipeInstructions', 'Calories']

# Define transformations for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Apply transformations
transformed_data = preprocessor.fit_transform(data)
transformed_columns = numerical_columns + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out())

# Combine with necessary text data and Calories
data_preprocessed = pd.DataFrame(transformed_data.toarray(), columns=transformed_columns, index=data.index)
for column in necessary_columns:
    data_preprocessed[column] = data[column]

# Save the processed data if necessary
data_preprocessed.to_csv('processed_meal_data.csv', index=False)
