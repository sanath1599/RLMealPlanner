import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_q_table(filename='q_table.csv'):
    """
    Load the Q-table from a CSV file.

    Args:
        filename (str): The name of the CSV file.

    Returns:
        numpy.ndarray: The loaded Q-table.
    """
    try:
        return pd.read_csv(filename, header=None).values
    except FileNotFoundError:
        return None

def save_q_table(q_table, filename='q_table.csv'):
    """
    Save the Q-table to a CSV file.

    Args:
        q_table (numpy.ndarray): The Q-table to be saved.
        filename (str): The name of the CSV file.
    """
    pd.DataFrame(q_table).to_csv(filename, header=False, index=False)

def save_last_meals(meals, filename='last_meals.csv'):
    """
    Save the list of last consumed meals to a CSV file.

    Args:
        meals (list): The list of last consumed meals.
        filename (str): The name of the CSV file.
    """
    pd.DataFrame({'RecipeId': meals}).to_csv(filename, index=False)

def load_last_meals(filename='last_meals.csv'):
    """
    Load the list of last consumed meals from a CSV file.

    Args:
        filename (str): The name of the CSV file.

    Returns:
        list: The list of last consumed meals.
    """
    try:
        df = pd.read_csv(filename)
        return df['RecipeId'].tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return []

def get_user_feedback(recommended_meals):
    """
    Get user feedback on recommended meals.

    Args:
        recommended_meals (pandas.DataFrame): The recommended meals.

    Returns:
        list: The user feedback for each recommended meal.
    """
    feedback = []
    for _, meal in recommended_meals.iterrows():
        print(f"Rate the meal: {meal['Name']} - Like (1), Neutral (0), Dislike (-1):")
        response = input()
        feedback.append(int(response))
    return feedback

def request_new_meal():
    """
    Ask the user if they want a new meal recommendation.

    Returns:
        int: The user's response (1 for Yes, 0 for No).
    """
    print("Would you like to get a new meal recommendation (Yes = 1 / No = 0)?")
    response = input()
    return int(response)

def calculate_meal_calories(num_meals, total_calories):
    """
    Calculate the distribution of calories for each meal based on the total number of meals and total calories.

    Args:
        num_meals (int): The total number of meals.
        total_calories (float): The total number of calories.

    Returns:
        list: A list of floats representing the distribution of calories for each meal.
    """
    if num_meals == 1:
        return [total_calories]
    elif num_meals == 2:
        return [total_calories * 0.7, total_calories * 0.3]
    else:
        calories = [total_calories * 0.4, total_calories * 0.3] + [total_calories * 0.3 / (num_meals - 2)] * (num_meals - 2)
        return calories

def recommend_meals(data, q_table, num_meals, last_meals, meal_calories):
    """
    Recommends meals based on the given data, Q-table, number of meals, last meals, and meal calories.

    Parameters:
    - data (pandas.DataFrame): The data containing information about the meals.
    - q_table (numpy.ndarray): The Q-table used for meal recommendations.
    - num_meals (int): The number of meals to recommend.
    - last_meals (list): The list of IDs of the last meals consumed.
    - meal_calories (list): The list of target calorie values for each meal.

    Returns:
    - recommended_meals (pandas.DataFrame): The recommended meals as a DataFrame.
    - meal_ids (list): The list of IDs of the recommended meals.
    """
    recommendations = []
    meal_ids = []

    valid_data = data[~data['RecipeId'].isin(last_meals)].copy()
    for i, calories in enumerate(meal_calories):
        valid_data['calorie_diff'] = abs(valid_data['Calories'] - calories)
        valid_indices = valid_data.sort_values(by='calorie_diff').index.tolist()

        if not valid_indices:
            valid_indices = list(data.index)  

        recommended_index = np.argsort(-q_table[valid_indices, :].mean(axis=0))[0]
        recommended_meal = valid_data.loc[valid_indices].iloc[0]  
        recommendations.append(recommended_meal)
        meal_ids.append(recommended_meal['RecipeId'])

        
        valid_data.drop(valid_indices[0], inplace=True)

    recommended_meals = pd.concat(recommendations, axis=1).T
    return recommended_meals, meal_ids


data = pd.read_csv('processed_meal_data.csv')
tfidf_columns = [col for col in data.columns if col.startswith('ingr_')]
feature_matrix = data[tfidf_columns].values
similarity_matrix = cosine_similarity(feature_matrix)

q_table = load_q_table()
last_5_meals = load_last_meals()

total_calories = float(input("Enter your target total calories for the day: "))
num_meals = int(input("How many meals do you want to plan for today? "))
meal_calories = calculate_meal_calories(num_meals, total_calories)
"""
Main loop for recommending meals and gathering user feedback.
    
Returns:
None
"""
while True:
    recommended_meals, meal_ids = recommend_meals(data, q_table, num_meals, last_5_meals, meal_calories)
    print("Recommended Meals for Today:")
    print(recommended_meals[['Name', 'Calories', 'Description']])

    if request_new_meal():
        continue
    else:
        feedback = get_user_feedback(recommended_meals)
        for recipe_id, meal_feedback in zip(meal_ids, feedback):
            idx = data.index[data['RecipeId'] == recipe_id].tolist()[0]
            adjustment = 10 if meal_feedback == 1 else -10 if meal_feedback == -1 else 0
            q_table[:, idx] += adjustment

        if len(last_5_meals) >= 10:
            last_5_meals = last_5_meals[-4:]  
        last_5_meals.extend(meal_ids)  

        save_last_meals(last_5_meals)
        save_q_table(q_table)
        break

print("Thank you for your feedback!")
