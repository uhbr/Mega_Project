import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress, pearsonr

# Load the dataset
dataset = pd.read_csv("datasets/parkinsons_updrs.data.csv")
df = dataset.copy()

# Step 1: Define custom time bins to categorize 'test_time' into weeks
min_time = df["test_time"].min()
max_time = df["test_time"].max()

# Create bins for time (every 7 days, with labels like 'Week 1', 'Week 2', etc.)
week_bins = np.arange(0, np.floor(max_time) + 7, 7)  # 7-day intervals
week_labels = [f"Week {i+1}" for i in range(len(week_bins)-1)]  # Labels for each week

# Assign each 'test_time' value to the corresponding week
df["week"] = pd.cut(df["test_time"], bins=week_bins, labels=week_labels, include_lowest=True)

# Step 2: Count how many subjects are in each week
subject_counts_per_week = df.groupby("week")["subject#"].nunique().reset_index()
subject_counts_per_week.columns = ["week", "num_subjects"]

# Print the number of subjects per week
print("Number of subjects per week:")
print(subject_counts_per_week)

# Step 3: Calculate the Pearson correlation coefficient between 'week' and 'test_time'
# Convert week labels to integers for correlation calculation
df["week_int"] = df["week"].cat.codes  # Convert 'week' labels into numeric values for correlation

# Calculate Pearson correlation
corr, _ = pearsonr(df["week_int"], df["test_time"])

# Perform linear regression to get the line
slope, intercept, _, _, _ = linregress(df["week_int"], df["test_time"])

# Generate the regression line (y = mx + b)
x_vals = np.linspace(df["week_int"].min(), df["week_int"].max(), 100)
y_vals = slope * x_vals + intercept

# Step 4: Sort weeks correctly (numeric order)
df["week_int"] = df["week_int"].astype(int)  # Ensure that 'week_int' is in integer format
sorted_weeks = sorted(df["week_int"].unique())  # Sort the weeks in numeric order
sorted_week_labels = [f"Week {i+1}" for i in sorted_weeks]  # Create corresponding week labels

# Step 5: Scatter plot using Plotly with 'week' and 'test_time', color by 'subject#'
fig = px.scatter(df, x="week", y="test_time", color="subject#", 
                 title=f"Test Time vs Week (Correlation: {corr:.2f})", 
                 labels={"week": "Week", "test_time": "Test Time", "subject#": "Subject"},
                 category_orders={"week": sorted_week_labels})  # Ensure week order is correct

# Add the regression line to the plot
fig.add_trace(go.Scatter(x=sorted_week_labels, y=y_vals, mode="lines", name="Regression Line", line={"color": "red"}))

# Show the plot
fig.show()
