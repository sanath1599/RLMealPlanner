import numpy as np
import pandas as pd

def load_q_table(filename):
    try:
        return pd.read_csv(filename).values
    except FileNotFoundError:
        return None

def load_feedback(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        return pd.DataFrame()

def update_q_table_from_feedback(q_table, feedback_df, data):
    if q_table is None:
        n_states = len(data)
        q_table = np.zeros((n_states, n_states))  # Initialize if not found
    if not feedback_df.empty:
        for _, feedback in feedback_df.iterrows():
            meal_id = feedback['MealID']
            liked = feedback['Liked']
            adjustment = 10 if liked == 1 else -10
            q_table[:, meal_id] += adjustment
    return q_table

def get_user_feedback(recommended_meals):
    feedback = {}
    for index, row in recommended_meals.iterrows():
        response = input(f"Did you like the meal '{row['Name']}' (1 for like, 0 for dislike)? ")
        feedback[index] = int(response)
    return feedback

data = pd.read_csv('processed_meal_data.csv')
q_table = load_q_table('q_table_updated.csv')
feedback_df = load_feedback('user_feedback.csv')
q_table = update_q_table_from_feedback(q_table, feedback_df, data)

# User inputs for personalized meal planning
target_calories = float(input("Enter your target calories per meal: "))
num_meals = int(input("How many meals do you want to plan for today? "))

# Q-Learning Parameters
learning_rate = 0.1
discount_factor = 0.95
epsilon = 0.2  # Higher epsilon for more exploration

# Initialize Q-table if not loaded
if q_table is None:
    n_states = len(data)
    q_table = np.zeros((n_states, n_states))

# Training block using Q-learning
for episode in range(100):  # Number of episodes can be adjusted based on performance and time
    state = np.random.randint(0, len(data))
    for _ in range(100):  # Steps per episode
        if np.random.rand() < epsilon:
            action = np.random.randint(0, len(data))
        else:
            action = np.argmax(q_table[state])
        actual_calories = data.loc[action, 'Calories']
        reward = -abs(actual_calories - target_calories)  # Reward based on calorie closeness
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
                                 learning_rate * (reward + discount_factor * np.max(q_table[action]))
        state = action

# Selection and recommendation logic
sorted_indices = np.argsort(-q_table.mean(axis=0))  # Sort by average Q-value across states
recommended_indices = []
count = 0

for idx in sorted_indices:
    if abs(data.loc[idx, 'Calories'] - target_calories) < 100 and count < num_meals:  # Close to calorie target
        recommended_indices.append(idx)
        count += 1

recommended_meals = data.iloc[recommended_indices]
print("Recommended Meals for Today:")
print(recommended_meals[['Name', 'Calories', 'Description']])

# Collect and save feedback
new_feedback = get_user_feedback(recommended_meals)
feedback_df = pd.DataFrame(list(new_feedback.items()), columns=['MealID', 'Liked'])
feedback_df.to_csv('user_feedback.csv', index=False)

# Save updated Q-table
pd.DataFrame(q_table).to_csv('q_table_updated.csv', index=False)
