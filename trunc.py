import pandas as pd

# Load only the first 10,000 rows from the CSV file
recipes = pd.read_csv('recipes.csv', nrows=10000)

# Save the truncated data to a new CSV file named 'recipes_trunc.csv'
recipes.to_csv('recipes_trunc.csv', index=False)

print("Truncated data saved to recipes_trunc.csv")
