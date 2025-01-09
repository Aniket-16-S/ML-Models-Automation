import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import sys

# Function to load data
def load_data():
    try:
        file_path = input("Enter the path to your CSV file: ").strip()
        data = pd.read_csv(file_path)
        print("File imported successfully!\n")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit()

# Function to display dataset information
def display_info(data):
    rows, cols = data.shape
    print(f"Dataset contains {rows} rows and {cols} columns.\n")
    print(data.head(6))
    print("\nColumn names:")
    for i, col in enumerate(data.columns):
        print(f"{i}: {col}")
    return rows

# Function to handle missing values
def handle_missing_values(data, col_name):
    print("\nChoose a strategy to handle missing values:")
    print("1. Mean\n2. Median\n3. Most Frequent\n4. Forward Fill (ffill)\n5. Backward Fill (bfill)\n6. Delete rows with missing values")
    while True:
        try:
            choice = int(input("Enter your choice (1-6): "))
            if choice in range(1, 7):
                break
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

    imputer = None
    if choice == 1:
        imputer = SimpleImputer(strategy="mean")
    elif choice == 2:
        imputer = SimpleImputer(strategy="median")
    elif choice == 3:
        imputer = SimpleImputer(strategy="most_frequent")
    elif choice == 4:
        data[col_name] = data[col_name].ffill()
    elif choice == 5:
        data[col_name] = data[col_name].bfill()
    elif choice == 6:
        data.dropna(subset=[col_name], inplace=True)

    if imputer:
        data[col_name] = imputer.fit_transform(data[[col_name]])
    print("Missing values handled successfully.")

# Function to preprocess data
def preprocess_data(data):
    target_col = None
    while True:
        try:
            target_col = int(input("Enter the index of the target (dependent) variable: "))
            if target_col in range(data.shape[1]):
                break
            else:
                print("Invalid column index. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

    base_col = None
    while True:
        try:
            base_col = int(input("Enter the index of the base (independent) variable: "))
            if base_col in range(data.shape[1]):
                break
            else:
                print("Invalid column index. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

    target_name = data.columns[target_col]
    base_name = data.columns[base_col]

    if data[base_name].isnull().any():
        handle_missing_values(data, base_name)
    if data[target_name].isnull().any():
        handle_missing_values(data, target_name)

    return data[[base_name]], data[[target_name]], base_name, target_name

# Function to train the model
def train_model(x, y):
    while True:
        try:
            test_s = float(input("Enter test size as a fraction(e.g., 0.2) : "))
            if 0 < test_s < 1:
                break
            else:
                if test_s > 1 and test_s < 100 :
                    test_s = float(test_s/100)
                else:
                    print("Please enter a value between 0 and 1.")
        except ValueError:
            print("Please enter a valid number.")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_s, random_state=42)

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model trained successfully. R-squared: {r2:.4f}, Mean Squared Error: {mse:.4f}")
    return model, x_test, predictions

# Function to visualize the regression line
def visualize(x_test, predictions, base_name, target_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test, predictions, color='blue', label='Data Points')
    plt.plot(x_test, predictions, color='red', label='Regression Line')
    plt.title('Simple Linear Regression Model')
    plt.xlabel(base_name)
    plt.ylabel(target_name)
    plt.legend()
    plt.grid()
    plt.show()

# Function to make predictions on new data
def predict_new(model, base_name):
    user_input = input("Do you want to make a prediction for new data? (yes/no): ").strip().lower()
    if user_input == 'yes':
        new_values = input(f"Enter values for {base_name} separated by commas: ").strip().split(',')
        try:
            new_values = [float(value) for value in new_values]
            new_df = pd.DataFrame({base_name: new_values})
            predictions = model.predict(new_df)
            for value, prediction in zip(new_values, predictions):
                print(f"Prediction for {base_name} = {value}: {prediction}")
        except ValueError:
            print("Invalid input. Please enter numerical values only.")

# Main function
def run():
    data = load_data()
    display_info(data)
    x, y, base_name, target_name = preprocess_data(data)
    model, x_test, predictions = train_model(x, y)
    visualize(x_test, predictions, base_name, target_name)
    predict_new(model, base_name)
    return
if __name__ == "__run__":
    run()
