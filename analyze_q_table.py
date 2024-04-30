import pandas as pd
import numpy as np

def load_q_table(filename='q_table.csv'):
    try:
        return pd.read_csv(filename, header=None)
    except FileNotFoundError:
        print("File not found. Ensure the Q-table file exists in the specified path.")
        return None

def analyze_q_table(q_table):
    # Basic stats
    print("Basic Statistics of Q-values:")
    print(q_table.describe())

    # Meals with the highest average Q-values
    highest_avg_q_values = q_table.mean(axis=0).sort_values(ascending=False)
    top_meals_indices = highest_avg_q_values.head(5).index
    print("\nTop 5 meals with the highest average Q-values:")
    print(highest_avg_q_values.head(5))

    # Most frequently best action
    most_freq_best_action = q_table.idxmax(axis=1).value_counts().head(5)
    print("\nTop 5 most frequently chosen meals as best action:")
    print(most_freq_best_action)

    return top_meals_indices

# Load the Q-table
q_table = load_q_table()

if q_table is not None:
    # Analyze the Q-table
    top_meals_indices = analyze_q_table(q_table)

    # Optionally, load meal data if available to show actual meal names
    try:
        meal_data = pd.read_csv('processed_meal_data.csv')
        print("\nDetails of Top Meals by Average Q-value:")
        print(meal_data.iloc[top_meals_indices][['Name', 'Calories', 'Description']])
    except FileNotFoundError:
        print("Meal data file not found. Skipping detailed meal output.")
