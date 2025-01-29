import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split


def preprocess_and_split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Preprocess the dataset by separating features and target variable, then split into train and test."""
    x = df.drop(columns=["name", "status"])  # Drop 'name' and 'status' columns
    y = df["status"]  # 'status' is the target variable (0 for healthy, 1 for PD)
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_and_evaluate_model(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> RandomForestClassifier:
    """Train the model, make predictions, evaluate, and print the classification report."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Print accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    return model, y_test, y_pred


def plot_precision_recall_curve(y_test: pd.Series, y_pred_prob: pd.DataFrame):
    """Plot the Precision-Recall curves for both Class 0 and Class 1 (Healthy and Parkinson's Disease)."""
    # Precision-Recall curve for class 1 (Parkinson's Disease)
    precision_1, recall_1, _ = precision_recall_curve(y_test, y_pred_prob[:, 1])

    # Precision-Recall curve for class 0 (Healthy)
    precision_0, recall_0, _ = precision_recall_curve(y_test, y_pred_prob[:, 0])

    # Create a plot for both class 0 and class 1 Precision-Recall curves
    plt.figure(figsize=(8, 6))
    plt.plot(recall_1, precision_1, color="b", label="Class 1 (Parkinson's Disease)")
    plt.fill_between(recall_1, precision_1, color="blue", alpha=0.2)
    plt.plot(recall_0, precision_0, color="r", label="Class 0 (Healthy)")
    plt.fill_between(recall_0, precision_0, color="red", alpha=0.2)

    # Labels and title
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Class 0 (Healthy) and Class 1 (Parkinson's Disease)")
    plt.legend(loc="best")
    plt.grid(True)

    # Show the plot
    plt.show()


def plot_feature_importance(model: RandomForestClassifier, x: pd.DataFrame):
    """Plots the feature importance of the Random Forest model as percentages."""
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": x.columns, "Importance": feature_importances})  # x.columns is already iterable
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Convert the importance to percentages
    importance_df["Importance"] = importance_df["Importance"] * 100

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importance for Detecting Parkinson's Disease (Percentage)")
    plt.xlabel("Importance (%)")
    plt.ylabel("Feature")
    plt.show()




def operating(df: pd.DataFrame):
    """Main function that ties together the machine learning process."""
    # Preprocess data and split into train/test
    x_train, x_test, y_train, y_test = preprocess_and_split_data(df)

    # Train Random Forest model and evaluate
    model, y_test, y_pred = train_and_evaluate_model(x_train, y_train, x_test, y_test)

    # Get predicted probabilities for both classes (0: healthy, 1: Parkinson's)
    y_pred_prob = model.predict_proba(x_test)  # Get probabilities for both class 0 and class 1

    # Plot Precision-Recall curves for both classes
    plot_precision_recall_curve(y_test, y_pred_prob)

    # Plot feature importance (with percentage scaling)
    plot_feature_importance(model, x_train)  # Pass x_train directly


if __name__ == "__main__":
    # Load the dataset and run the operating function
    dataset = pd.read_csv("datasets/parkinsons.data")
    operating(dataset)
