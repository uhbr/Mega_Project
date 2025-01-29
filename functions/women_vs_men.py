import pandas as pd
import plotly.graph_objects as go
from scipy import stats


def compare_gender_trends_with_pvalues(df):
    """Compare the trends of male and female subjects for all numerical parameters in the dataframe
    and display p-values for each parameter in a Plotly bar chart showing only `age`, `motor_UPDRS`, and `total_UPDRS`.

    df: DataFrame containing the dataset
    gender_column: Column name containing gender information (0=male, 1=female)

    Assumptions:
    1. The `sex` column in the dataset contains the gender information, where `0` denotes male subjects and `1` denotes female subjects.
    2. The dataset contains numerical columns representing various Parkinson’s disease-related parameters, including `age`, `motor_UPDRS`, and `total_UPDRS`.
    3. Missing values are handled at the parameter level by removing rows with `NaN` values for the numerical columns before performing t-tests.
    4. A p-value threshold of 0.05 is used to determine statistical significance. P-values less than this threshold indicate a significant difference between males and females.
    5. Only the fixed parameters `age`, `motor_UPDRS`, and `total_UPDRS` are currently compared for gender-based differences.
    6. The dataset is expected to have the following structure:
        - A `sex` column indicating gender (0 for male, 1 for female).
        - A variety of numerical columns that represent Parkinson’s disease-related parameters.
        - Optional columns such as `index` or `subject#`, which are dropped before performing calculations.
    7. The function relies on Plotly to display an interactive bar chart, requiring the `plotly` library.

    Limitations:
    - No adjustment (such as Bonferroni correction) is made for multiple comparisons, which may increase the risk of Type I errors.
    - The t-test assumes that the data for each gender is approximately normally distributed. If this assumption is violated, alternative tests (e.g., Mann-Whitney U test) may be more appropriate.
    """
    # Step 1: Drop 'index' and 'subject#' columns (if they exist) from the calculations
    df_clean = df.drop(columns=["index", "subject#"], errors="ignore")

    # Step 2: Select all numerical columns for comparison
    numerical_columns = df_clean.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Step 3: Group by gender to split the dataset
    df_male = df_clean[df_clean["sex"] == 0]
    df_female = df_clean[df_clean["sex"] == 1]

    # Step 4: Perform t-tests for each parameter and collect p-values
    p_values = {}
    for param in numerical_columns:
        male_values = df_male[param]
        female_values = df_female[param]
        t_stat, p_value = stats.ttest_ind(male_values, female_values, equal_var=False)
        p_values[param] = p_value

    # Step 5: Print significant and non-significant parameters
    significant_parameters = [param for param, p_val in p_values.items() if p_val < 0.05]
    non_significant_parameters = [param for param, p_val in p_values.items() if p_val >= 0.05]

    # Print all numerical parameters and their p-values
    print("Significant Parameters:")
    for param in significant_parameters:
        print(f"{param}'s p-value = {p_values[param]:.3f}")

    print("\nNon-Significant Parameters:")
    print(non_significant_parameters)

    # Step 6: Plot the graph for the following fixed parameters: 'age', 'motor_UPDRS', and 'total_UPDRS'
    graph_parameters = ["age", "motor_UPDRS", "total_UPDRS"]
    graph_parameters = [param for param in graph_parameters if param in df_clean.columns]

    # Step 8: Create a Plotly bar chart to display male and female values for age, motor_UPDRS, and total_UPDRS
    fig = go.Figure()

    # Add bars for each parameter (male and female)
    for param in graph_parameters:
        male_mean = df_male[param].mean()  # Get scalar value for mean (averaged per subject)
        female_mean = df_female[param].mean()  # Get scalar value for mean (averaged per subject)

        # Add male data
        fig.add_trace(go.Bar(
            x=[param],
            y=[male_mean],
            name=f"Male - {param}",
            marker={"color": "blue"},
            text="Male",
            textposition="auto",
            hoverinfo="text",
        ))

        # Add female data
        fig.add_trace(go.Bar(
            x=[param],
            y=[female_mean],
            name=f"Female - {param}",
            marker={"color": "red"},
            text="Female",
            textposition="auto",
            hoverinfo="text",
        ))

        # Check if the difference is significant (p-value < 0.05)
        p_val = p_values[param]
        if p_val < 0.05:
            # Add a star for significant results
            fig.add_annotation(
                x=param, y=max(male_mean, female_mean) + 1,  # Place the star above the bar
                text="*",
                font={"size": 15, "color": "black"},
                showarrow=False,  # No arrow for star
                align="center"
            )

            # Add p-value next to the star
            fig.add_annotation(
                x=param, y=max(male_mean, female_mean) + 0.5,  # Position below the star
                text=f"p-value = {p_val:.3f}",
                font={"size": 12},
                showarrow=False  # No arrow for p-value annotation
            )

    # Customize the layout of the plot
    fig.update_layout(
        title="Comparison of Mean Values by Gender (age, motor_UPDRS, total_UPDRS)",
        xaxis_title="Parameter",
        yaxis_title="Mean Value",
        template="plotly_white",  # Using a clean template
        height=600,
        showlegend=False,
        xaxis={"tickmode": "array"},
    )

    # Show the plot
    fig.show()


if __name__ == "__main__":
    df = pd.read_csv("datasets/parkinsons_updrs.data.csv")

    # Group by gender and compare trends
    compare_gender_trends_with_pvalues(df)
