import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
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
            target_col = int(input("Enter the index of the target (dependent) variable column: "))
            if 0 <= target_col < data.shape[1]:
                break
            else:
                print("Invalid column index. Try again.")
        except ValueError:
            print("Please enter a valid integer.")

    base_cols = None
    while True:
        try:
            base_cols = input("Enter the indices of the base (independent) variable columns, separated by commas: ").strip()
            base_cols = [int(index) for index in base_cols.split(',')]
            if all(0 <= col < data.shape[1] for col in base_cols):
                break
            else:
                print("Invalid column indices. Try again.")
        except ValueError:
            print("Please enter valid integers separated by commas.")

    target_name = data.columns[target_col]
    base_names = [data.columns[col] for col in base_cols]

    for base_name in base_names:
        if data[base_name].isnull().any():
            handle_missing_values(data, base_name)
    if data[target_name].isnull().any():
        handle_missing_values(data, target_name)

    return data[base_names], data[[target_name]], base_names, [target_name]

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

    model = LogisticRegression()
    model.fit(x_train, y_train.values.ravel())
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Model trained successfully. Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, predictions))
    return model, x_test, predictions

# Function to visualize the results
def visualize(x_test, predictions, base_names, target_names):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(predictions)), predictions, label='Predictions', color='blue', alpha=0.6)
    plt.title('Logistic Regression Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Class')
    plt.legend()
    plt.grid()
    plt.show()

# Function to make predictions on new data
def predict_new(model, base_names):
    user_input = input("Do you want to make a prediction for new data? (yes/no): ").strip().lower()
    if user_input == 'yes':
        new_values = input(f"Enter values for {', '.join(base_names)} separated by commas for each row, separated by semicolons: ").strip()
        try:
            rows = [list(map(float, row.split(','))) for row in new_values.split(';')]
            new_df = pd.DataFrame(rows, columns=base_names)
            predictions = model.predict(new_df)
            for i, row in enumerate(rows):
                print(f"Prediction for input {row}: {predictions[i]}")
        except ValueError:
            print("Invalid input. Please enter numerical values only.")

# Main function
def run():
    data = load_data()
    display_info(data)
    x, y, base_names, target_names = preprocess_data(data)
    model, x_test, predictions = train_model(x, y)
    visualize(x_test, predictions, base_names, target_names)
    predict_new(model, base_names)
    return 
if __name__ == "__run__":
    run()
