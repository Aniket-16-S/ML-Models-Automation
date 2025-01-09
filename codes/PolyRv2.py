import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
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
    # Detect string columns
    string_cols = data.select_dtypes(include=['object']).columns.tolist()

    if string_cols:
        print("\nThe dataset contains categorical (string) columns:")
        for col in string_cols:
            print(f"- {col}")

        # Ask user if they want to encode these columns
        encode_choice = input("Do you want to encode these columns using one-hot encoding? (yes/no): ").strip().lower()
        if encode_choice == 'yes':
            # Apply OneHotEncoding
            print("Encoding string columns...")
            column_transformer = ColumnTransformer(
                transformers=[
                    ('onehot', OneHotEncoder(), string_cols)
                ], remainder='passthrough'
            )
            data = pd.DataFrame(
                column_transformer.fit_transform(data),
                columns=column_transformer.get_feature_names_out()
            )
            print("Encoding completed.")
            print("Updated column names:", list(data.columns))  # Debugging info

    # Updated data and column references must match the transformed data
    while True:
        try:
            print("Columns available:", list(data.columns))  # Debugging info
            target_cols = input("Enter the indices of the target (dependent) variable columns, separated by commas: ").strip()
            target_cols = [int(index) for index in target_cols.split(',')]
            if all(0 <= col < data.shape[1] for col in target_cols):
                break
            else:
                print("Invalid column indices. Try again.")
        except ValueError:
            print("Please enter valid integers separated by commas.")

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

    target_names = [data.columns[col] for col in target_cols]
    base_names = [data.columns[col] for col in base_cols]

    for base_name in base_names:
        if data[base_name].isnull().any():
            handle_missing_values(data, base_name)
    for target_name in target_names:
        if data[target_name].isnull().any():
            handle_missing_values(data, target_name)

    return data[base_names], data[target_names], base_names, target_names

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
        except ValueError as v:
            print(f"Please enter a valid number. {v} ")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_s, random_state=42)

    degre = int(input("Enter the degree of the polynomial features: ").strip())
    poly = PolynomialFeatures(degree=degre)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    model = LinearRegression()
    model.fit(x_train_poly, y_train)
    predictions = model.predict(x_test_poly)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model trained successfully. R-squared: {r2:.4f}, Mean Squared Error: {mse:.4f}")
    return model, poly, x_test, predictions

# Function to visualize the regression
def visualize(poly, x_test, predictions, base_names, target_names):
    plt.figure(figsize=(10, 6))
    for i, target_name in enumerate(target_names):
        plt.scatter(poly.transform(x_test)[:, 1], predictions[:, i], label=f'{target_name} Predictions')
    plt.title('Polynomial Regression Model')
    plt.xlabel(', '.join(base_names))
    plt.ylabel(', '.join(target_names))
    plt.legend()
    plt.grid()
    plt.show()

# Function to make predictions on new data
def predict_new(model, poly, base_names):
    user_input = input("Do you want to make a prediction for new data? (yes/no): ").strip().lower()
    if user_input == 'yes':
        new_values = input(f"Enter values for {', '.join(base_names)} separated by commas for each row, separated by semicolons: ").strip()
        try:
            rows = [list(map(float, row.split(','))) for row in new_values.split(';')]
            new_df = pd.DataFrame(rows, columns=base_names)
            new_df_poly = poly.transform(new_df)
            predictions = model.predict(new_df_poly)
            for i, row in enumerate(rows):
                print(f"Prediction for input {row}: {predictions[i]}")
        except ValueError:
            print("Invalid input. Please enter numerical values only.")

# Main function
def run():
    data = load_data()
    display_info(data)
    x, y, base_names, target_names = preprocess_data(data)
    model, poly, x_test, predictions = train_model(x, y)
    visualize(poly, x_test, predictions, base_names, target_names)
    predict_new(model, poly, base_names)
    return
if __name__ == "__run__":
    run()
run()
