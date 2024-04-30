import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_q_table(filename):
    try:
        return pd.read_csv(filename, header=None).values
    except FileNotFoundError:
        return None

def save_q_table(q_table, filename='q_table.csv'):
    pd.DataFrame(q_table).to_csv(filename, header=False, index=False)

def get_user_feedback(recommended_meals):
    feedback = []
    for _, meal in recommended_meals.iterrows():
        print(f"Do you like this meal: {meal['Name']} (Yes = 1 / No = 0)?")
        response = input()
        feedback.append(int(response))
    return feedback

def request_new_meal():
    print("Would you like to get a new meal recommendation (Yes = 1 / No = 0)?")
    response = input()
    return int(response)

def generate_recommendations(data, q_table, num_meals):
    # This function could be enhanced to consider total calories or other factors
    recommended_indices = np.argsort(-q_table.mean(axis=0))[:num_meals]
    recommended_meals = data.iloc[recommended_indices]
    additional_meal_index = np.random.randint(0, len(data))
    additional_meal = data.iloc[[additional_meal_index]]
    recommended_meals = pd.concat([recommended_meals, additional_meal], ignore_index=True)
    return recommended_meals

# Load the processed data including TF-IDF features for ingredients
data = pd.read_csv('processed_meal_data.csv')
tfidf_columns = [col for col in data.columns if col.startswith('ingr_')]

# Generate feature matrix and compute similarity matrix from TF-IDF features
feature_matrix = data[tfidf_columns].values
similarity_matrix = cosine_similarity(feature_matrix)

# Load or initialize Q-table
q_table = load_q_table('q_table.csv')
if q_table is None or q_table.shape[0] != len(data):
    q_table = np.zeros((len(data), len(data)))  # Initialize a new Q-table if none or dimension mismatch

# Collect user preferences
total_calories = float(input("Enter your target total calories for the day: "))
num_meals = int(input("How many meals do you want to plan for today? "))

# Q-learning parameters
learning_rate = 0.1
discount_factor = 0.95
epsilon = 0.1  # Exploration rate

# Main recommendation and feedback loop
while True:
    # Recommend meals
    recommended_meals = generate_recommendations(data, q_table, num_meals)
    print("Recommended Meals for Today:")
    print(recommended_meals[['Name', 'Calories', 'Description']])
    
    # Check if user wants new meals
    if request_new_meal():
        print("Generating new recommendations...")
        continue
    else:
        # Collect feedback on the recommended meals
        feedback = get_user_feedback(recommended_meals)
        
        # Update Q-table based on actual user feedback
        for idx, like in zip(recommended_meals.index, feedback):
            q_table[:, idx] += (10 if like == 1 else -10)  # Increase or decrease the value based on feedback
        
        save_q_table(q_table)
        break

print("Thank you for your feedback!")
