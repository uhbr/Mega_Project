import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import numpy as np

def plot_parkinsons_progression(df, min_subjects = 21):
    """This function plots the progression of different Parkinson's parameters over time (in weeks).
    The dataset is filtered to include only weeks with a minimum number of subjects.

    Parameters:
        df (pd.DataFrame): The dataset containing Parkinson's data.
        min_subjects (int): The minimum number of subjects per week to include. Default is 21.

    Returns:
        None: Displays the progression plots for the mean patient.
    """
    # Step 1: Convert 'test_time' to weeks (round down to the nearest integer for week number)
    df["week"] = np.floor(df["test_time"] / 7).astype(int)

    # Step 2: Count the unique subjects for each week
    subjects_per_week = df.groupby("week")["subject#"].nunique()

    # Step 3: Filter out weeks with fewer than `min_subjects` subjects
    valid_weeks = subjects_per_week[subjects_per_week >= min_subjects].index

    # Step 4: Filter the original dataframe to include only the valid weeks
    df_filtered = df[df["week"].isin(valid_weeks)]

    # Step 5: Define parameters for the three groups of plots
    parameters_motor_updrs = ["motor_UPDRS", "total_UPDRS", "HNR"]
    parameters_dfa_rpde = ["DFA", "PPE", "Shimmer(dB)", "RPDE"]
    parameters_rest = [
        "Jitter(Abs)", "Jitter:RAP", "Jitter(%)", "Jitter:PPQ5", "Jitter:DDP",
        "Shimmer", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", "Shimmer:DDA",
        "NHR"
    ]

    # Calculate the mean for each subject at each week (in case there are multiple tests in the same week)
    subject_week_means = df_filtered.groupby(["subject#", "week"])[parameters_motor_updrs + parameters_dfa_rpde
     + parameters_rest].mean().reset_index()

    # Step 6: Calculate the mean across all subjects for each week
    mean_patient_weekly = subject_week_means.groupby("week")[parameters_motor_updrs + parameters_dfa_rpde
     + parameters_rest].mean().reset_index()

    # Step 7: Create subplots with 3 graphs
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    # Plot for the first set of parameters (motor_UPDRS, total_UPDRS, HNR)
    for param in parameters_motor_updrs:
        axs[0].plot(mean_patient_weekly["week"], mean_patient_weekly[param], label=param)

    # Plot for the second set of parameters (DFA, PPE, Shimmer(dB), RPDE)
    for param in parameters_dfa_rpde:
        axs[1].plot(mean_patient_weekly["week"], mean_patient_weekly[param], label=param)

    # Plot for the third set of parameters (the rest)
    for param in parameters_rest:
        axs[2].plot(mean_patient_weekly["week"], mean_patient_weekly[param], label=param)

    # Customize each subplot
    axs[0].set_title("Motor UPDRS, Total UPDRS, HNR")
    axs[0].set_xlabel("Week")
    axs[0].set_ylabel("Parameter Value")
    axs[0].legend(loc="best")
    axs[0].tick_params(axis="x", rotation=45)
    axs[0].grid(True)  # Add gridlines for better readability

    axs[1].set_title("DFA, PPE, Shimmer(dB), RPDE")
    axs[1].set_xlabel("Week")
    axs[1].set_ylabel("Parameter Value")
    axs[1].legend(loc="best")
    axs[1].tick_params(axis="x", rotation=45)
    axs[1].grid(True)  # Add gridlines for better readability

    axs[2].set_title("Other Parameters")
    axs[2].set_xlabel("Week")
    axs[2].set_ylabel("Parameter Value")
    axs[2].legend(loc="best")
    axs[2].tick_params(axis="x", rotation=45)
    axs[2].grid(True)  # Add gridlines for better readability

    # Adjust the layout for a better fit
    plt.tight_layout()

    # Step 8: Display the plot
    plt.show()



if __name__ == "__main__":
    dataset = pd.read_csv("datasets/parkinsons_updrs.data.csv")
    # Copy the dataset to avoid modifying the original
    df = dataset.copy()

    # Plot progression
    plot_parkinsons_progression(df)


