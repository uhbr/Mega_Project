import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_correlation_matrix(df, min_subjects=21):
    """Let's visualize how different Parkinson's disease parameters are related to each other over time!
    This function calculates and plots the correlation matrix, giving us insights into how the parameters (like voice features, disease progression, etc.) correlate across different weeks.

    Parameters:
        df (pd.DataFrame): The dataframe containing Parkinson's data (I’m assuming it has the same structure as mine with 'test_time', 'week', and 'subject#' columns).
        min_subjects (int): The minimum number of subjects needed for a week to ensure the data's validity. We don’t want any weeks with too few subjects messing up the results. Default is 21.

    Returns:
        None: Displays a heatmap showing how the different parameters are correlated with each other over time.

    Assumptions:
        - The dataset includes a 'test_time' column which tracks the time of the test for each subject.
        - The dataset includes a 'subject#' column to uniquely identify each subject.
        - The dataset might not have a 'week' column initially, so this function will create it by grouping the 'test_time' into weeks.
    """
    # Step 1: Calculate weeks starting from test_time = 0 (if week is not already present)
    if "week" not in df.columns:
        df["week"] = (df["test_time"] // 7).astype(int)  # Grouping test_time into weeks

    # Step 2: Filter out weeks with fewer than `min_subjects` subjects
    subjects_per_week = df.groupby("week")["subject#"].nunique()
    valid_weeks = subjects_per_week[subjects_per_week >= min_subjects].index

    df_filtered = df[df["week"].isin(valid_weeks)]

    # Step 3: Select only numerical columns for correlation
    numerical_columns = df_filtered.select_dtypes(include=[float, int]).columns.tolist()

    # Remove 'week' from numerical columns for correlation (we'll keep it as a separate factor)
    if "week" in numerical_columns:
        numerical_columns.remove("week")

    numerical_columns.remove("index")
    numerical_columns.remove("subject#")

    # Step 4: Group by 'week' and calculate the mean for each parameter
    mean_patient_weekly = df_filtered.groupby("week")[numerical_columns].mean().reset_index()

    # Step 5: Calculate the correlation matrix for all numerical parameters
    corr_matrix = mean_patient_weekly[numerical_columns].corr()

    # Step 6: Plot the correlation matrix as a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="PiYG", fmt=".2f", vmin=-1, vmax=1)

    # Customize the plot with labels and title
    plt.title("Correlation Matrix for All Numerical Parameters (Including Week)", fontsize=16)
    plt.xlabel("Parameters", fontsize=12)
    plt.ylabel("Parameters", fontsize=12)

    # Rotate the x-axis and y-axis labels for better visibility
    plt.xticks(rotation=90)

    # Adjust layout for better fit
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_correlation_matrix_status(df):
    """Let’s take a look at how the numerical features in our dataset relate to each other!
    This function plots a correlation matrix for all numerical parameters, giving us a quick overview of their relationships.

    Parameters:
        df (pd.DataFrame): The dataframe with the data to analyze (again, assuming the structure is the same as mine, with numerical features to correlate).

    Returns:
        None: Displays a heatmap of the correlation matrix between the numerical parameters.

    Assumptions:
        - The dataset is expected to have numerical features to calculate correlations (like jitter, shimmer, etc.).
        - We’re assuming all columns that are numeric in the dataset are relevant for the correlation matrix.
    """
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=["number"])

    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="RdBu", fmt=".2f")

    # Add title and show plot
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df2 = pd.read_csv("datasets/parkinsons.data")
    plot_correlation_matrix_status(df2)
    dataset = pd.read_csv("datasets/parkinsons_updrs.data.csv")
    df = dataset.copy()
    df2 = pd.read_csv("datasets/parkinsons.data")
    df2_c = df2.copy()
    # Plot correlation matrix
    plot_correlation_matrix(df)
