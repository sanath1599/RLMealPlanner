import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def load_q_table(filename='q_table.csv', num_actions=None):
    """
    Loads the Q-table from a CSV file or creates a new Q-table if the file is not found.

    Args:
        filename (str, optional): The name of the CSV file to load the Q-table from. Defaults to 'q_table.csv'.
        num_actions (int, optional): The number of actions in the Q-table. Required if the Q-table file is not found.

    Returns:
        numpy.ndarray or None: The loaded Q-table as a NumPy array, or None if the Q-table file is not found and num_actions is not provided.
    """
    try:
        return pd.read_csv(filename, header=None).values
    except FileNotFoundError:
        if num_actions is not None:
            return np.zeros((num_actions, num_actions))
        else:
            print("Error: num_actions must be provided if Q-table file is not found.")
            return None


def epsilon_greedy_policy(q_values, state, epsilon=0.1):
    """
    Implements an epsilon-greedy policy for selecting actions based on Q-values.

    Parameters:
    - q_values (numpy.ndarray): The Q-values for each action in each state.
    - state (int): The current state.
    - epsilon (float): The exploration rate. Defaults to 0.1.

    Returns:
    - action (int): The selected action.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(0, q_values.shape[1])
    else:
        return np.argmax(q_values[state])

def update_q_values(q_table, state, action, reward, next_state, alpha=0.1, gamma=0.95):
    """
    Update the Q-values in the Q-table based on the given state, action, reward, and next state.

    Parameters:
    - q_table (numpy.ndarray): The Q-table storing the Q-values for each state-action pair.
    - state (int): The current state.
    - action (int): The action taken in the current state.
    - reward (float): The reward received for taking the action in the current state.
    - next_state (int): The next state after taking the action.
    - alpha (float, optional): The learning rate. Defaults to 0.1.
    - gamma (float, optional): The discount factor. Defaults to 0.95.

    Returns:
    - None
    """
    best_next_action = np.argmax(q_table[next_state])
    q_table[state, action] += alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])

def load_last_meals(filename='last_meals.csv'):
    """
    Load the last meals from a CSV file.

    Args:
        filename (str): The name of the CSV file to load the meals from. Defaults to 'last_meals.csv'.

    Returns:
        list: A list of RecipeIds representing the last meals.

    Raises:
        FileNotFoundError: If the specified file is not found.
        pd.errors.EmptyDataError: If the specified file is empty.

    """
    try:
        df = pd.read_csv(filename)
        return df['RecipeId'].tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return []

def save_last_meals(meals, filename='last_meals.csv'):
    """
    Save the list of meals to a CSV file.

    Parameters:
    meals (list): A list of recipe IDs representing the meals.
    filename (str): The name of the CSV file to save the meals. Default is 'last_meals.csv'.

    Returns:
    None
    """
    pd.DataFrame({'RecipeId': meals}).to_csv(filename, index=False)

def save_q_table(q_table, filename='q_table.csv'):
    """
    Save the Q-table to a CSV file.

    Parameters:
    - q_table (numpy.ndarray): The Q-table to be saved.
    - filename (str): The name of the CSV file to save the Q-table to. Default is 'q_table.csv'.
    """
    pd.DataFrame(q_table).to_csv(filename, header=False, index=False)

def get_user_feedback(recommended_meals):
    """
    Collects user feedback for recommended meals.

    Parameters:
    - recommended_meals (pandas.DataFrame): A DataFrame containing the recommended meals.

    Returns:
    - feedback (list): A list of user feedback for each recommended meal. Each feedback is represented as an integer:
                       - Like: 1
                       - Neutral: 0
                       - Dislike: -1
    """
    feedback = []
    for _, meal in recommended_meals.iterrows():
        print(f"Rate the meal: {meal['Name']} - Like (1), Neutral (0), Dislike (-1):")
        response = input()
        feedback.append(int(response))
    return feedback

def request_new_meal():
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
        list: A list of calorie values for each meal.

    Raises:
        None
    """
    if num_meals == 1:
        return [total_calories]
    elif num_meals == 2:
        return [total_calories * 0.6, total_calories * 0.4]
    else:
        return [total_calories * 0.35, total_calories * 0.3] + [total_calories * 0.35 / (num_meals - 2)] * (num_meals - 2)

def recommend_meals(data, q_table, num_meals, last_meals, meal_calories):
    """
    Recommends a list of meals based on the given data, Q-table, number of meals, last meals, and meal calories.

    Parameters:
        data (DataFrame): The data containing information about all available meals.
        q_table (ndarray): The Q-table used for meal recommendation.
        num_meals (int): The number of meals to recommend.
        last_meals (list): The list of IDs of the last meals consumed.
        meal_calories (list): The list of target calorie values for each meal.

    Returns:
        recommended_meals (DataFrame): The recommended meals as a DataFrame.
        meal_ids (list): The list of IDs of the recommended meals.
    """
    recommendations = []
    meal_ids = []
    valid_data = data[~data['RecipeId'].isin(last_meals)].copy()
    for i, calories in enumerate(meal_calories):
        valid_data['calorie_diff'] = abs(valid_data['Calories'] - calories)
        valid_indices = valid_data.sort_values(by='calorie_diff').index.tolist()
        valid_indices = [idx for idx in valid_indices if idx < q_table.shape[0]]

        if not valid_indices:
            valid_indices = list(data.index)[:q_table.shape[0]] 
        
        recommended_indices = np.argsort(-q_table[valid_indices, :].mean(axis=0))[:num_meals]
        recommended_meals = valid_data.iloc[recommended_indices]
        recommendations.append(recommended_meals)
        meal_ids.extend(recommended_meals['RecipeId'].tolist())
        # valid_data.drop(recommended_indices, inplace=True)
    recommended_meals = pd.concat(recommendations)[:num_meals]
    return recommended_meals, meal_ids


def compare_q_tables(original_q_table, new_q_table):
    """
    Compare two Q-tables by plotting their mean Q-values for each recipe index.

    Parameters:
    original_q_table (numpy.ndarray): The original Q-table.
    new_q_table (numpy.ndarray): The new Q-table.

    Returns:
    None
    """

    plt.figure(figsize=(10, 6))

    # Plot original Q-table
    plt.plot(original_q_table.mean(axis=0), label='Original Q-table', color='blue')

    # Plot new Q-table
    plt.plot(new_q_table.mean(axis=0), label='New Q-table', color='red')

    plt.title('Comparison of Q-tables')
    plt.xlabel('Recipe Index')
    plt.ylabel('Mean Q-value')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_q_table(q_table):
    """
    Plots the mean Q-value over time for the given Q-table.

    Parameters:
    q_table (numpy.ndarray): The Q-table containing Q-values for each state-action pair.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(q_table.mean(axis=0))
    plt.title('Mean Q-value Over Time')
    plt.xlabel('Recipe Index')
    plt.ylabel('Mean Q-value')
    plt.grid(True)
    plt.show()

def plot_calorie_differences(valid_data):
    """
    Plot the distribution of calorie differences.

    Parameters:
    valid_data (DataFrame): A DataFrame containing the valid data.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    plt.hist(valid_data['calorie_diff'], bins=20, edgecolor='k')
    plt.title('Distribution of Calorie Differences')
    plt.xlabel('Calorie Difference')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_user_feedback(feedback):
    """
    Plots the distribution of user feedback.

    Parameters:
    feedback (list): A list of user feedback values.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    plt.hist(feedback, bins=[-1, 0, 1, 2], edgecolor='k')
    plt.title('User Feedback Distribution')
    plt.xlabel('Feedback')
    plt.ylabel('Frequency')
    plt.xticks([-1, 0, 1], ['Dislike', 'Neutral', 'Like'])
    plt.grid(True)
    plt.show()

data = pd.read_csv('processed_meal_data.csv')
tfidf_columns = [col for col in data.columns if col.startswith('ingr_')]
feature_matrix = data[tfidf_columns].values
similarity_matrix = cosine_similarity(feature_matrix)
num_actions = data.shape[0] 
q_table = load_q_table('q_table.csv', num_actions)
q_table_orig= q_table.copy()

last_5_meals = load_last_meals()
total_calories = float(input("Enter your target total calories for the day: "))
num_meals = int(input("How many meals do you want to plan for today? "))
meal_calories = calculate_meal_calories(num_meals, total_calories)
while True:
    recommended_meals, meal_ids = recommend_meals(data, q_table, num_meals, last_5_meals, meal_calories)
    print("Recommended Meals for Today:")
    print(recommended_meals[['Name', 'Calories', 'RecipeInstructions']])
    # plot_calorie_differences(recommended_meals)
    if len(last_5_meals) >= 10:
            last_5_meals = last_5_meals[-4:]
    last_5_meals.extend(meal_ids)
    
    save_last_meals(last_5_meals)
    load_last_meals()
    if request_new_meal():
        continue
    else:
        feedback = get_user_feedback(recommended_meals)
        plot_user_feedback(feedback)
        for recipe_id, meal_feedback in zip(meal_ids, feedback):
            idx = data.index[data['RecipeId'] == recipe_id].tolist()[0]
            adjustment = 10 if meal_feedback == 1 else -10 if meal_feedback == -1 else 0
            q_table[:, idx] += adjustment
        plot_q_table(q_table)
        save_q_table(q_table)
        compare_q_tables(q_table_orig, q_table)
        break
print("Thank you for your feedback!")
