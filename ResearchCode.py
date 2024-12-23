# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from google.colab import drive
# drive.mount('/content/drive')

# Load the CSV file
df = pd.read_csv('D:/Work File/Project Files/drugDataAnalysis-Research/Dataset/drugdata.csv')

#------------------------------------------------------

#The full dataset is shown.
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None) # Show all columns
print(df)

#------------------------------------------------------
#Each column of the data set is shown.
print(df.columns)
#------------------------------------------------------
# Display the first few rows
print(df.head(12))
#------------------------------------------------------
# Display the first few rows
print(df.tail(12))
#------------------------------------------------------
# Check column names and data types
print(df.info())
#------------------------------------------------------
# Summary statistics
df.describe()
#------------------------------------------------------
# Summary statistics
df[['Yaba','Phensedyl','Cannabis','Buprenorphine','Alcohol','Sedative, Hypnotic, Tranquilizer','Injecting Drug','Poly drugs']].describe()
#------------------------------------------------------
# Check for missing values
print(df.isnull().sum())
#------------------------------------------------------
#fill missing values:
df.fillna(0, inplace=True)
# Check for missing values AFTER imputation
print("\nMissing values after imputation:\n", df.isnull().sum())
#------------------------------------------------------


#!pip install plotly # Install Plotly library and this line used for Google Colab
import plotly.express as px

# Convert percentage strings to numerical values
drug_list = ['Heroin', 'Yaba', 'Phensedyl', 'Cannabis', 'Buprenorphine', 'Alcohol', 'Sedative, Hypnotic, Tranquilizer', 'Injecting Drug', 'Poly drugs', 'Other']
for drug in drug_list:
    # Replace 'Null' with NaN before removing '%' and converting to float
    df[drug] = df[drug].replace('Null', pd.NA)  # Replace 'Null' with NaN (Not a Number)
    # Ensure the column is of string type before applying str operations
    df[drug] = df[drug].astype(str).str.rstrip('%').astype(float, errors='ignore') # convert to strings first

# Melt the DataFrame to create a long-form dataset for Plotly
df_melted = df.melt(id_vars=['Year'], value_vars=drug_list, var_name='Drug', value_name='Percentage')

# Create an interactive line plot using Plotly Express
fig = px.line(df_melted, x='Year', y='Percentage', color='Drug', title='Trends of Drug Usage Over Years')
fig.show()
#------------------------------------------------------

# Convert percentage strings to numerical values
drug_list = ['Heroin', 'Yaba', 'Phensedyl', 'Cannabis', 'Buprenorphine', 'Alcohol', 'Sedative, Hypnotic, Tranquilizer', 'Injecting Drug', 'Poly drugs', 'Other']
for drug in drug_list:
    # Replace 'Null' with NaN before removing '%' and converting to float
    df[drug] = df[drug].replace('Null', pd.NA)  # Replace 'Null' with NaN (Not a Number)
    # Ensure the column is of string type before applying str operations
    df[drug] = df[drug].astype(str).str.rstrip('%').astype(float, errors='ignore') # convert to strings first

# Create subplots for each drug
num_drugs = len(drug_list)
num_cols = 3  # Number of columns in the subplot grid
num_rows = (num_drugs + num_cols - 1) // num_cols  # Calculate number of rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True)
fig.suptitle('Trends of Drug Usage Over Years', fontsize=16)

for i, drug in enumerate(drug_list):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    ax.plot(df['Year'], df[drug], label=drug)
    ax.set_title(drug)
    ax.set_ylabel('Percentage')
    ax.grid(True)

# Remove empty subplots if any
for i in range(num_drugs, num_rows * num_cols):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent overlapping
plt.show()
#------------------------------------------------------

# List of factors influencing drug addiction
factors_list = [
    'Curiosity', 'Influence of Friends', 'Mental Disorder', 'Adverse Family Environtment', 'Easy Available','Unemployment', 'Frustration', 'Ignore About Consequence', 'Medical Hazards','Desire to get easy pleasure']

# Melt the DataFrame to create a long-form dataset
df_factors_melted = df.melt(id_vars=['Year'], value_vars=factors_list,
                            var_name='Factor', value_name='Percentage')

# Convert 'Percentage' column to numeric, handling errors
df_factors_melted['Percentage'] = pd.to_numeric(df_factors_melted['Percentage'].str.rstrip('%'), errors='coerce')

# Group by factor to calculate the average percentage across all years
df_factors_avg = df_factors_melted.groupby('Factor', as_index=False)['Percentage'].mean()

# Create an interactive bar chart using Plotly Express
fig_factors = px.bar(
    df_factors_avg,
    x='Factor',
    y='Percentage',
    color='Factor',
    title='Factors Influencing Drug Addiction',
    labels={'Percentage': 'Average Percentage'},
    text='Percentage'
)

# Customize appearance
fig_factors.update_layout(xaxis_tickangle=-45)  # Tilt factor labels for readability
fig_factors.show()
#------------------------------------------------------

# List of factors influencing drug addiction
factors_list = [
    'Curiosity', 'Influence of Friends', 'Mental Disorder',
    'Adverse Family Environtment', 'Easy Available', 'Unemployment',
    'Frustration', 'Ignore About Consequence', 'Medical Hazards',
    'Desire to get easy pleasure'
]

# Ensure percentages are numeric, handling non-string columns
for factor in factors_list:
    # Check if the column dtype is object (likely string) before applying str operations
    if df[factor].dtype == object:
        df[factor] = pd.to_numeric(df[factor].str.rstrip('%'), errors='coerce')
    else:
        # If already numeric or other type, no need to convert
        pass

# Create subplots for each factor
num_factors = len(factors_list)
num_cols = 3  # Number of columns in the subplot grid
num_rows = (num_factors + num_cols - 1) // num_cols  # Calculate number of rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True)
fig.suptitle('Trends of Factors Influencing Drug Addiction Over Years', fontsize=16)

for i, factor in enumerate(factors_list):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    ax.plot(df['Year'], df[factor], label=factor)
    ax.set_title(factor)
    ax.set_ylabel('Percentage')
    ax.grid(True)

# Remove empty subplots if any
for i in range(num_factors, num_rows * num_cols):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent overlapping
plt.show()
#------------------------------------------------------

# List of age groups
age_groups = ['Up to 15', '16 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 40 ', '41 to 45', '46 to 50', '51 to Above']

# Melt the DataFrame to create a long-form dataset
df_age_melted = df.melt(id_vars=['Year'], value_vars=age_groups,
                        var_name='Age Group', value_name='Percentage')

# Convert 'Percentage' column to numeric, handling errors and possibly split concatenated percentages
# Split the concatenated percentages assuming they are separated by '%'
df_age_melted['Percentage'] = df_age_melted['Percentage'].str.split('%').str[0]

# Convert to numeric, handling errors
df_age_melted['Percentage'] = pd.to_numeric(df_age_melted['Percentage'], errors='coerce')


# Group by age group to calculate the average percentage across all years
df_age_avg = df_age_melted.groupby('Age Group', as_index=False)['Percentage'].mean()

# Create an interactive pie chart using Plotly Express
fig_age = px.pie(
    df_age_avg,
    names='Age Group',
    values='Percentage',
    title='Age Group Distribution of Drug Addiction',
    hole=0.4  # Optional: creates a donut chart
)

# Customize appearance
fig_age.update_traces(textinfo='percent+label')  # Show percentages and labels
fig_age.show()
#------------------------------------------------------

# List of age groups
age_groups = [
    'Up to 15', '16 to 20', '21 to 25', '26 to 30',
    '31 to 35', '36 to 40 ', '41 to 45', '46 to 50', '51 to Above'
]

# Ensure percentages are numeric
for age_group in age_groups:
    df[age_group] = pd.to_numeric(df[age_group].str.rstrip('%'), errors='coerce')

# Create subplots for each age group
num_age_groups = len(age_groups)
num_cols = 3  # Number of columns in the subplot grid
num_rows = (num_age_groups + num_cols - 1) // num_cols  # Calculate number of rows

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=True)
fig.suptitle('Trends of Age Group Distribution Over Years', fontsize=16)

for i, age_group in enumerate(age_groups):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]
    ax.plot(df['Year'], df[age_group], label=age_group)
    ax.set_title(age_group)
    ax.set_ylabel('Percentage')
    ax.grid(True)

# Remove empty subplots if any
for i in range(num_age_groups, num_rows * num_cols):
    fig.delaxes(axes.flatten()[i])

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent overlapping
plt.show()
#------------------------------------------------------

# Compute the correlation matrix, only including numeric columns
correlation = df.select_dtypes(include=np.number).corr()

# Plot a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
#------------------------------------------------------

from IPython.display import HTML
import base64

def create_download_link(df, title = "Download the Cleaned Dataset", filename = "cleaned_drug_addiction_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '''<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">
                <button class="download-button">{title}</button>
            </a>'''
    html = html.format(payload=payload,title=title,filename=filename)

    # CSS styling for the button with hover, animation, and running line
    css_style = """
    <style>
    .download-button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }

    .download-button:hover {
        background-color: #3e8e41;
    }

    .download-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border: 2px solid #fff;
        border-radius: 5px;
        box-sizing: border-box; /* Ensure borders are within the element */
        animation: running-line 2s linear infinite;
    }

    @keyframes running-line {
        0% { clip-path: inset(0 100% 100% 0); } /* Start at top-left */
        25% { clip-path: inset(0 0 100% 0); } /* Move to top-right */
        50% { clip-path: inset(0 0 0 0); } /* Move to bottom-right */
        75% { clip-path: inset(0 100% 0 0); } /* Move to bottom-left */
        100% { clip-path: inset(0 100% 100% 0); } /* Back to top-left */
    }
    </style>
    """

    return HTML(css_style + html)

create_download_link(df)
#------------------------------------------------------

from sklearn.linear_model import LinearRegression

# List of categories to predict
categories = ['Heroin', 'Yaba', 'Phensedyl', 'Cannabis',
              'Buprenorphine', 'Alcohol', 'Sedative, Hypnotic, Tranquilizer',
              'Injecting Drug', 'Poly drugs', 'Other']

# Loop through each category and perform predictions
for category in categories:
    # Select data for the current category
    data = df[['Year', category]].dropna()  # Ensure no missing values

    # Convert the category column to numeric, handling errors
    data[category] = pd.to_numeric(data[category], errors='coerce')

    # Drop rows with NaN values after conversion
    data = data.dropna()

    # Prepare features (X) and target (y)
    X = data['Year'].values.reshape(-1, 1)  # Independent variable
    y = data[category].values  # Dependent variable

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict for 2030
    future_years = np.arange(min(data['Year']), 2031).reshape(-1, 1)
    predicted_values = model.predict(future_years)

    # Print the prediction for 2030
    prediction_2030 = model.predict([[2030]])[0]
    print(f"Predicted {category} usage in 2030: {prediction_2030:.2f}%")

    # Plot the data and prediction
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Year'], y, color='blue', label='Actual Data')
    plt.plot(future_years, predicted_values, color='red', label='Trend Line')
    plt.scatter([2030], [prediction_2030], color='green', label=f'Prediction (2030)')

    # Formatting the X-axis to show every year
    plt.xticks(np.arange(min(data['Year']), 2031, 1), rotation=45)  # Show every year
    plt.title(f"Trend Prediction for {category}")
    plt.xlabel('Year')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to avoid overlapping
    plt.show()
#------------------------------------------------------

# First, remove the 'Month' column if it exists
if 'Month' in df.columns:
    df.drop('Month', axis=1, inplace=True)  # axis=1 indicates dropping a column

# Convert the percentage columns to numeric values (remove '%', convert to float)
for column in df.columns[2:]:  # Skipping 'Year', 'Region' columns
    # Check if the column dtype is object (likely string) before applying str.replace
    if df[column].dtype == object:
        df[column] = pd.to_numeric(df[column].str.replace('%', ''), errors='coerce')  # Remove '%' and convert to float

# Replace any NaN values (which were originally 'Null' or '<NA>') with 0
df.fillna(0, inplace=True)

# Create a MultiIndex with all combinations of Year and Region
all_years = df['Year'].unique()
all_regions = df['Region'].unique()

# Create MultiIndex from all combinations of Year and Region
multi_index = pd.MultiIndex.from_product([all_years, all_regions], names=['Year', 'Region'])

# Group by 'Year' and 'Region', summing the values of each drug/factor
grouped_data = df.groupby(['Year', 'Region']).sum()

# Reindex to ensure every combination of Year and Region is included, filling missing combinations with 0
grouped_data = grouped_data.reindex(multi_index, fill_value=0)

# Reset the index to make 'Year' and 'Region' columns again
grouped_data.reset_index(inplace=True)

# Ensure 'Region' values retain their original format from the dataset
grouped_data['Region'] = grouped_data['Region'].astype(str)


# Sort the data by 'Year'
grouped_data.sort_values(by='Year', inplace=True)

# Display the first few rows of the cleaned data
print(grouped_data.head())
#------------------------------------------------------

#showing the dataset by per year
print(grouped_data)
#------------------------------------------------------

from sklearn.linear_model import LinearRegression

# Initialize a dictionary to store predictions for each column
predictions_2030 = {}

# Select columns to predict (excluding 'Year' and 'Region')
columns_to_predict = grouped_data.columns[2:]

# Loop through each column and predict for 2030
for column in columns_to_predict:
    # Extract year and values for the current column
    X = grouped_data['Year'].values.reshape(-1, 1)  # Independent variable (Year)
    y = grouped_data[column].values  # Dependent variable (e.g., Heroin, Yaba)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict for 2030
    predicted_value = model.predict([[2030]])[0]
    predictions_2030[column] = predicted_value

# Convert predictions into a DataFrame for easier analysis
predictions_df = pd.DataFrame.from_dict(predictions_2030, orient='index', columns=['Predicted Percentage (2030)'])
predictions_df.sort_values(by='Predicted Percentage (2030)', ascending=False, inplace=True)

# Display predictions
print(predictions_df)


##------------------------------------------------------

# Plot the predicted percentages for 2030
predictions_df.plot(kind='bar', figsize=(12, 6), color='skyblue', legend=False)
plt.title('Predicted Drug/Factor Percentages for 2030')
plt.ylabel('Predicted Percentage')
plt.xlabel('Drug/Factor')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
##------------------------------------------------------

# List of drug columns
drug_columns = ['Heroin', 'Yaba', 'Phensedyl', 'Cannabis', 'Buprenorphine', 'Alcohol',
                'Sedative, Hypnotic, Tranquilizer', 'Injecting Drug', 'Poly drugs', 'Other']

# Initialize dictionary to store drug predictions
drug_predictions = {}

for drug in drug_columns:
    # Extract year and values for the current drug
    X = grouped_data['Year'].values.reshape(-1, 1)
    y = grouped_data[drug].values

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict for 2030
    predicted_value = model.predict([[2030]])[0]
    drug_predictions[drug] = predicted_value

# Convert predictions into DataFrame
drug_predictions_df = pd.DataFrame.from_dict(drug_predictions, orient='index', columns=['Predicted Percentage (2030)'])
drug_predictions_df.sort_values(by='Predicted Percentage (2030)', ascending=False, inplace=True)

# Display predictions for drugs
print("Drug Predictions for 2030:")
print(drug_predictions_df)

##------------------------------------------------------

#Graph visualization of drug prediction results
drug_predictions_df.plot(kind='bar', figsize=(12, 6), color='orange', legend=False)
plt.title('Predicted Drug Usage Percentages for 2030')
plt.ylabel('Predicted Percentage')
plt.xlabel('Drugs')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
##------------------------------------------------------

# List of factor columns
factor_columns = ['Curiosity', 'Influence of Friends', 'Mental Disorder', 'Adverse Family Environtment',
                  'Easy Available', 'Unemployment', 'Frustration', 'Ignore About Consequence',
                  'Medical Hazards', 'Desire to get easy pleasure']

# Initialize dictionary to store factor predictions
factor_predictions = {}

for factor in factor_columns:
    # Extract year and values for the current factor
    X = grouped_data['Year'].values.reshape(-1, 1)
    y = grouped_data[factor].values

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict for 2030
    predicted_value = model.predict([[2030]])[0]
    factor_predictions[factor] = predicted_value

# Convert predictions into DataFrame
factor_predictions_df = pd.DataFrame.from_dict(factor_predictions, orient='index', columns=['Predicted Percentage (2030)'])
factor_predictions_df.sort_values(by='Predicted Percentage (2030)', ascending=False, inplace=True)

# Display predictions for factors
print("Factor Predictions for 2030:")
print(factor_predictions_df)

##------------------------------------------------------

#Graph visualization of factor prediction results
factor_predictions_df.plot(kind='bar', figsize=(12, 6), color='green', legend=False)
plt.title('Predicted Factors Influencing Drug Addiction for 2030')
plt.ylabel('Predicted Percentage')
plt.xlabel('Factors')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
##------------------------------------------------------

# List of age group columns
age_group_columns = ['Up to 15', '16 to 20', '21 to 25', '26 to 30', '31 to 35',
                     '36 to 40 ', '41 to 45', '46 to 50', '51 to Above']

# Initialize dictionary to store age group predictions
age_predictions = {}

for age_group in age_group_columns:
    # Extract year and values for the current age group
    X = grouped_data['Year'].values.reshape(-1, 1)
    y = grouped_data[age_group].values

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict for 2030
    predicted_value = model.predict([[2030]])[0]
    age_predictions[age_group] = predicted_value

# Convert predictions into DataFrame
age_predictions_df = pd.DataFrame.from_dict(age_predictions, orient='index', columns=['Predicted Percentage (2030)'])
age_predictions_df.sort_values(by='Predicted Percentage (2030)', ascending=False, inplace=True)

# Display predictions for age groups
print("Age Group Predictions for 2030:")
print(age_predictions_df)

##------------------------------------------------------

#Graph visualization of age prediction results
age_predictions_df.plot.pie(y='Predicted Percentage (2030)', labels=age_predictions_df.index, autopct='%1.1f%%', figsize=(8, 8))
plt.title('Predicted Age Group Distribution for 2030')
plt.ylabel('')
plt.show()
##------------------------------------------------------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Dictionary to store the accuracy metrics
accuracy_metrics = {}

# Loop through each column and calculate accuracy
for column in columns_to_predict:
    # Extract year and values for the current column
    X = grouped_data['Year'].values.reshape(-1, 1)  # Independent variable (Year)
    y = grouped_data[column].values  # Actual values

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions on training data for evaluation
    y_pred = model.predict(X)

    # Calculate accuracy metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # Store the results in the dictionary
    accuracy_metrics[column] = {
        'R-squared': r2,
        'Mean Absolute Error': mae,
        'Root Mean Squared Error': rmse
    }

# Display the accuracy metrics
accuracy_df = pd.DataFrame(accuracy_metrics).T
print(accuracy_df)
##------------------------------------------------------

# Convert the DataFrame to percentages for easier visualization
accuracy_df['R-squared (%)'] = accuracy_df['R-squared'] * 100
accuracy_df['Mean Absolute Error'] = accuracy_df['Mean Absolute Error']
accuracy_df['Root Mean Squared Error'] = accuracy_df['Root Mean Squared Error']

# Create a new DataFrame with only the required columns for plotting
accuracy_plot_df = accuracy_df[['R-squared (%)', 'Mean Absolute Error', 'Root Mean Squared Error']]

# Set up the figure size and style
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")

# Plot the R-squared values as a bar graph
accuracy_plot_df['R-squared (%)'].plot(kind='bar', color='skyblue', label='R-squared (%)', alpha=0.7)
plt.ylabel('Percentage/Value')
plt.title('Model Accuracy Metrics by Column')

# Overlay MAE and RMSE as line plots for comparison
accuracy_plot_df['Mean Absolute Error'].plot(kind='line', marker='o', label='Mean Absolute Error', color='orange')
accuracy_plot_df['Root Mean Squared Error'].plot(kind='line', marker='s', label='Root Mean Squared Error', color='red')

# Add legend and labels
plt.xlabel('Columns')
plt.xticks(rotation=45, ha='right')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

##------------------------------------------------------

# Dictionary to store the accuracy percentages
accuracy_percentages = {}

# Loop through each column and calculate accuracy
for column in columns_to_predict:
    # Extract year and values for the current column
    X = grouped_data['Year'].values.reshape(-1, 1)  # Independent variable (Year)
    y = grouped_data[column].values  # Actual values

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions on training data for evaluation
    y_pred = model.predict(X)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, y_pred)

    # Calculate accuracy as a percentage
    mean_actual = np.mean(y)  # Mean of actual values
    accuracy = 1 - (mae / mean_actual)  # Accuracy calculation
    accuracy_percentages[column] = accuracy * 100  # Convert to percentage

# Convert accuracy percentages to a DataFrame for display
accuracy_df = pd.DataFrame.from_dict(accuracy_percentages, orient='index', columns=['Accuracy (%)'])
print(accuracy_df)
