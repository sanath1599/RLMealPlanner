import pandas as pd
import numpy as np

def load_q_table(filename='q_table.csv'):
    try:
        return pd.read_csv(filename, header=None)
    except FileNotFoundError:
        print("File not found. Ensure the Q-table file exists in the specified path.")
        return None

def analyze_q_table(q_table):
    print("Basic Statistics of Q-values:")
    print(q_table.describe())

    highest_avg_q_values = q_table.mean(axis=0).sort_values(ascending=False)
    print("\nTop 5 meals with the highest average Q-values:")
    print(highest_avg_q_values.head(5))

    lowest_avg_q_values = q_table.mean(axis=0).sort_values(ascending=True)
    print("\nTop 5 meals with the lowest average Q-values:")
    print(lowest_avg_q_values.head(5))

    return highest_avg_q_values.head(5).index, lowest_avg_q_values.head(5).index

q_table = load_q_table()
if q_table is not None:
    top_meals_indices, least_meals_indices = analyze_q_table(q_table)
    try:
        meal_data = pd.read_csv('processed_meal_data.csv')
        print("\nDetails of Top Meals by Average Q-value:")
        print(meal_data.iloc[top_meals_indices][['Name', 'Calories', 'Description']])
        print("\nDetails of Least Favored Meals by Average Q-value:")
        print(meal_data.iloc[least_meals_indices][['Name', 'Calories', 'Description']])
    except FileNotFoundError:
        print("Meal data file not found. Skipping detailed meal output.")
