import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


def predict_patient_progression(df, patient, years_ahead):
    """Predicts the progression of UPDRS for either the mean patient or a specific patient.

    df: DataFrame containing the dataset
    patient: 'mean' for mean patient or a patient ID (1-42) for a specific patient
    years_ahead: The number of years to predict (default is 10)

    Returns:
        None: Plots the predicted progression for the selected patient.
    """
    # Assumption: The 'week' column might not be there, so we create it from 'test_time' if necessary
    if "week" not in df.columns:
        if "test_time" in df.columns:
            df["week"] = (df["test_time"] / 7).astype(int)  # Create 'week' from 'test_time' (assuming it's in days)
        else:
            msg = "'week' column or 'test_time' column not found in the dataset."  # Assumed error handling
            raise ValueError(msg)

    # Initialize df_grouped which will be used for regression
    df_grouped = pd.DataFrame()

    if patient == "mean":
        # Calculate the mean parameters per week for the entire dataset
        df_grouped = df.groupby("week")[["motor_UPDRS", "total_UPDRS"]].mean()

        # Assumption: We filter out weeks with fewer than 21 patients for valid analysis
        subject_counts_per_week = df.groupby("week")["subject#"].nunique()
        valid_weeks = subject_counts_per_week[subject_counts_per_week >= 21].index
        df_grouped = df_grouped[df_grouped.index.isin(valid_weeks)]

    elif isinstance(patient, int) and 1 <= patient <= 42:
        # Assumption: If a specific patient is selected, we handle it gracefully
        subject_data = df[df["subject#"] == patient]

        if subject_data.empty:
            print(f"No data found for subject {patient}")
            return

        # Calculate the mean for the specific patient per week
        df_grouped = subject_data.groupby("week")[["motor_UPDRS", "total_UPDRS"]].mean()

    else:
        print("Invalid patient. Please enter 'mean' or a valid patient ID (1-42).")  # Assumption: patient input must be valid
        return

    # Prepare data for linear regression (assuming linearity in progression)
    X = df_grouped.index.values.reshape(-1, 1)  # weeks as input feature
    y_motor = df_grouped["motor_UPDRS"].values
    y_total = df_grouped["total_UPDRS"].values

    # Train models for motor and total UPDRS
    motor_model = LinearRegression()
    total_model = LinearRegression()

    motor_model.fit(X, y_motor)
    total_model.fit(X, y_total)

    # Predict progression over the next 10 years (weeks) - assuming linear trend for the future
    time_range = np.linspace(df_grouped.index.min(), df_grouped.index.max() + (years_ahead * 52), 100).reshape(-1, 1)
    predicted_motor = motor_model.predict(time_range)
    predicted_total = total_model.predict(time_range)

    # Plot the predicted progression along with original data points
    fig = go.Figure()

    # Add original data points (actual observed data by week)
    fig.add_trace(go.Scatter(
        x=df_grouped.index,
        y=df_grouped["motor_UPDRS"],
        mode="markers",
        name="Observed motor_UPDRS",
        marker={"color": "blue"},
        text="Observed data",
        hoverinfo="x+y+text",
    ))

    fig.add_trace(go.Scatter(
        x=df_grouped.index,
        y=df_grouped["total_UPDRS"],
        mode="markers",
        name="Observed total_UPDRS",
        marker={"color": "red"},
        text="Observed data",
        hoverinfo="x+y+text",
    ))

    # Add predicted motor and total UPDRS values
    fig.add_trace(go.Scatter(
        x=time_range.flatten(),
        y=predicted_motor,
        mode="lines",
        name=f"Predicted motor_UPDRS ({'Mean' if patient == 'mean' else f'Subject {patient}'})",
        line={"color": "blue"}
    ))

    fig.add_trace(go.Scatter(
        x=time_range.flatten(),
        y=predicted_total,
        mode="lines",
        name=f"Predicted total_UPDRS ({'Mean' if patient == 'mean' else f'Subject {patient}'})",
        line={"color": "red"}
    ))

    # Update layout and show the figure
    fig.update_layout(
        title=f"Predicted and Observed Progression of Parkinson's Disease ({'Mean' if patient == 'mean'
                                                                            else f'Subject {patient}'})",
        xaxis_title="Weeks",
        yaxis_title="UPDRS Score",
        template="plotly_white",
        height=600,
        showlegend=True,
    )

    fig.show()




if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("datasets/parkinsons_updrs.data.csv")

    # 1. Predict progression for the mean patient over the next 10 years
    predict_patient_progression(df, patient="mean", years_ahead=10)
