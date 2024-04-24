import pandas as pd

# Load the data
df = pd.read_csv('recipes.csv')

# Define Nutritional Categories
def categorize_nutritional_value(total_nutrition):
    if total_nutrition <= 200:
        return 'low'
    elif 200 < total_nutrition <= 400:
        return 'medium'
    else:
        return 'high'

# Apply categorization function to create 'NutritionalValue' column
df['NutritionalValue'] = df.apply(lambda row: categorize_nutritional_value(row['ProteinContent'] + row['SugarContent'] + row['FiberContent'] + row['CarbohydrateContent'] + row['SodiumContent'] + row['CholesterolContent'] + row['SaturatedFatContent'] + row['Calories'] + row['FatContent']), axis=1)

# Count the number of recipes in each nutritional category
nutritional_counts = df['NutritionalValue'].value_counts()

# Display the counts
print("Number of recipes with low nutritional value:", nutritional_counts['low'])
print("Number of recipes with medium nutritional value:", nutritional_counts['medium'])
print("Number of recipes with high nutritional value:", nutritional_counts['high'])


# Save the modified DataFrame to a new CSV file
df.to_csv('categorized_recipes.csv', index=False)
