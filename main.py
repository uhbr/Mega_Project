import tkinter as tk
from tkinter import messagebox, simpledialog

import pandas as pd
from functions.corr_matrix import plot_correlation_matrix, plot_correlation_matrix_status
from functions.detection import operating
from functions.predicting_progression import predict_patient_progression
from functions.progression_func import plot_parkinsons_progression
from functions.women_vs_men import compare_gender_trends_with_pvalues

# Load datasets globally (make sure your file paths are correct)
dataset1 = pd.read_csv("datasets/parkinsons_updrs.data.csv").copy()
dataset2 = pd.read_csv("datasets/parkinsons.data").copy()


def compare_gender_trends():
    compare_gender_trends_with_pvalues(dataset1)


def plot_correlation_matrix_func():
    plot_type = simpledialog.askstring("Input", "Which dataset would you like to see? 1-dataset1, 2-dataset2:")
    if plot_type == "1":
        plot_correlation_matrix(dataset1)
    elif plot_type == "2":
        plot_correlation_matrix_status(dataset2)
    else:
        messagebox.showerror("Invalid Input", "Invalid number!")


def check_datasets_info():
    messagebox.showinfo("Datasets Info", 
                        "Dataset1 has info about 42 Parkinson's patients that were tested across a few months.\n"
                        "Dataset2 has info about 32 subjects, 23 of them have Parkinson's.")


def train_and_evaluate():
    operating(dataset2)


def predict_progression():
    while True:
        # Prompt the user for patient ID
        patient = simpledialog.askstring("Input", "Enter the patient ID (1-42) or 'mean' for mean patient:")

        # If the user pressed cancel or closed the dialog
        if patient is None:
            return

        # Prompt the user for the number of years ahead to predict
        years_ahead = simpledialog.askinteger("Input", "Enter the number of years ahead to predict:")

        # If the user pressed cancel or closed the dialog
        if years_ahead is None:
            return

        # Handle the 'mean' input case
        if patient.lower() == "mean":
            patient = "mean"  # Convert the patient variable to 'mean' if it's entered
        else:
            # Try to convert the input to an integer if it's not 'mean'
            try:
                patient = int(patient)
                if not (1 <= patient <= 42):
                    messagebox.showerror("Invalid Input", "Please enter a valid patient ID between 1 and 42.")
                    continue  # Continue prompting for valid input
            except ValueError:
                # If the input is neither a valid integer nor 'mean', show an error
                messagebox.showerror("Invalid Input", "Please enter a valid patient ID (1-42) or 'mean'.")
                continue  # Continue prompting for valid input

        # Validate the years ahead input (must be a positive integer)
        if years_ahead <= 0:
            messagebox.showerror("Invalid Input", "Please enter a valid number of years ahead (positive number).")
            continue

        # Call the actual progression function with the validated inputs
        predict_patient_progression(dataset1, patient, years_ahead)

        # Ask the user if they'd like to try again
        again_choice = messagebox.askquestion("Try another patient?", "Would you like to try a different patient?")
        if again_choice.lower() != "yes":
            break  # Exit the loop if the user does not want to try another patient


def plot_progression():
    plot_parkinsons_progression(dataset1)


def exit_program(root):
    # Show a goodbye message before quitting
    if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
        messagebox.showinfo("Goodbye", "Thank you for using the Parkinson's Disease Analysis Menu. Goodbye!")
        root.quit()  # Close the application



def create_gui():
    """Creates a simple GUI menu for the user to interact with."""
    root = tk.Tk()
    root.title("Parkinson's Disease Analysis Menu")
    root.geometry("700x700")
    # Adding a title label inside the GUI
    title_label = tk.Label(root, text="Parkinson's Disease Analysis", font=("Arial", 16, "bold"))
    title_label.pack(pady=20)  # Add some padding to the label for spacing

    # Buttons for different functions
    tk.Button(root, text="Compare Gender Trends with P-values", command=compare_gender_trends).pack(pady=10)
    tk.Button(root, text="Plot Correlation Matrix", command=plot_correlation_matrix_func).pack(pady=10)
    tk.Button(root, text="Check the Datasets Info", command=check_datasets_info).pack(pady=10)
    tk.Button(root, text="Train and Evaluate Model", command=train_and_evaluate).pack(pady=10)
    tk.Button(root, text="Predict Patient Progression", command=predict_progression).pack(pady=10)
    tk.Button(root, text="Plot Parkinson's Disease Progression", command=plot_progression).pack(pady=10)
    tk.Button(root, text="Exit", command=lambda: exit_program(root)).pack(pady=10)

    # Start the Tkinter event loop to display the window
    root.mainloop()


if __name__ == "__main__":
    create_gui()
