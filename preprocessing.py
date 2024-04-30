import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def convert_iso8601_to_minutes(iso_str):
    if pd.isna(iso_str):
        return 0
    pattern = re.compile(r'P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?')
    parts = pattern.match(iso_str)
    days, hours, minutes = parts.groups(default='0')
    return int(days) * 1440 + int(hours) * 60 + int(minutes)

# Load data
data = pd.read_csv('recipes_trunc.csv')

# Convert time durations
for col in ['CookTime', 'PrepTime', 'TotalTime']:
    data[col] = data[col].apply(convert_iso8601_to_minutes)

# Extract and clean categories and ingredients using regex that captures strings in double quotes
def extract_text(column):
    return column.str.extractall(r'"([^"]+)"').groupby(level=0).agg(' '.join).reindex(column.index, fill_value='')

data['CombinedIngredients'] = extract_text(data['RecipeIngredientParts'])
data['RecipeCategory'] = extract_text(data['RecipeCategory'])

# Ensure all text data is string and non-empty for TF-IDF
data['CombinedIngredients'] = data['CombinedIngredients'].fillna('').astype(str)
data['RecipeCategory'] = data['RecipeCategory'].fillna('').astype(str)

# Only proceed with TF-IDF if there is any non-empty data
if data['RecipeCategory'].str.strip().eq('').all():
    print("Warning: No valid category data available for TF-IDF vectorization.")
else:
    tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    categories_tfidf = tfidf_vectorizer.fit_transform(data['RecipeCategory'])
    categories_df = pd.DataFrame(categories_tfidf.toarray(), columns=['cat_' + str(i) for i in range(categories_tfidf.shape[1])])
    data = pd.concat([data, categories_df], axis=1)

if data['CombinedIngredients'].str.strip().eq('').all():
    print("Warning: No valid ingredients data available for TF-IDF vectorization.")
else:
    tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    ingredients_tfidf = tfidf_vectorizer.fit_transform(data['CombinedIngredients'])
    ingredients_df = pd.DataFrame(ingredients_tfidf.toarray(), columns=['ingr_' + str(i) for i in range(ingredients_tfidf.shape[1])])
    data = pd.concat([data, ingredients_df], axis=1)

# Save processed data
data.to_csv('processed_meal_data.csv', index=False)
