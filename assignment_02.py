import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

def read_preprocess_and_transpose(filename):
    """
    Reads, preprocesses, and transposes a World Bank format CSV file.

    Parameters:
        filename (str): The name of the CSV file.

    Returns:
        pd.DataFrame: The DataFrame after preprocessing.
        pd.DataFrame: The transposed DataFrame.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename, encoding='latin-1', na_values=['..'])  # Adjust encoding if needed

        # Specify the columns to be removed
        columns_to_remove = ['Series Name', 'Series Code', 'Country Code', '2013 [YR2013]', '2014 [YR2014]',
                             '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]', '2021 [YR2021]', '2022 [YR2022]']

        # Remove the specified columns
        df = df.drop(columns=columns_to_remove, errors='ignore')

        # Remove rows with NaN values
        df = df.dropna()

        # Transpose the DataFrame
        transposed_df = df.set_index('Country Name').transpose()

        # Fill NaN values with the mean of each column
        transposed_df = transposed_df.apply(pd.to_numeric, errors='coerce')
        transposed_df = transposed_df.apply(lambda col: col.fillna(col.mean()))

        return df, transposed_df

    except Exception as e:
        print(f"Error reading or processing CSV file: {e}")
        return None, None

# Example usage:
filename = 'CO2.csv'  # Replace with the actual file name
df_CO2, df_transposed_CO2 = read_preprocess_and_transpose(filename)
df_energy_consumption, df_transposed_energy_consumption = read_preprocess_and_transpose('energy_consumption.csv')
df_arable_land, df_transposed_arable_land = read_preprocess_and_transpose('arable_land.csv')
df_forest_land, df_transposed_forest_land = read_preprocess_and_transpose('forest_land.csv')
df_GDP, df_transposed_GDP = read_preprocess_and_transpose('GDP.csv')
df_population, df_transposed_population = read_preprocess_and_transpose('population.csv')

def plot_bar_chart(df_transposed, title):
    """
    Plots a bar chart for each country over different years.

    Parameters:
    - df_transposed (pd.DataFrame): The transposed DataFrame containing indicator values.
    - title (str): The title of the plot.

    Returns:
    - None
    """
    # Extract the country names and years from the DataFrame
    countries = df_transposed.columns.tolist()
    years = df_transposed.index.tolist()

    # Set width of the bar
    barWidth = 0.10
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set position of the bar on X axis
    br = [np.arange(len(countries)) + i * barWidth for i in range(len(years))]

    # Make the plot
    for idx, year in enumerate(years):
        ax.bar(br[idx], df_transposed.loc[year], width=barWidth, edgecolor='grey', label=year)

    # Adding Xticks
    plt.xlabel('Country', fontweight='bold', fontsize=15)
    plt.ylabel('Indicator Value', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth * (len(years) / 2) for r in range(len(countries))], countries, rotation=45, ha='right')
    plt.title(title, fontsize=16)
    plt.legend()
    plt.show()

# Example usage:
# Assuming df_transposed_CO2 is your transposed DataFrame
plot_bar_chart(df_transposed_CO2, 'CO2 emission')

def plot_gdp_bar_chart(df_transposed, title):
    """
    Plots a bar chart for GDP for each country over different years.

    Parameters:
    - df_transposed (pd.DataFrame): The transposed DataFrame containing GDP values.
    - title (str): The title of the plot.

    Returns:
    - None
    """
    # Extract the country names and years from the DataFrame
    countries = df_transposed.columns.tolist()
    years = df_transposed.index.tolist()

    # Set width of the bar
    barWidth = 0.10
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set position of the bar on X axis
    br = [np.arange(len(countries)) + i * barWidth for i in range(len(years))]

    # Make the plot
    for idx, year in enumerate(years):
        ax.bar(br[idx], df_transposed.loc[year], width=barWidth, edgecolor='grey', label=year)

    # Adding Xticks
    plt.xlabel('Country', fontweight='bold', fontsize=15)
    plt.ylabel('GDP Value', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth * (len(years) / 2) for r in range(len(countries))], countries, rotation=45, ha='right')
    plt.title(title, fontsize=16)
    plt.legend()
    plt.show()

# Example usage:
# Assuming df_transposed_GDP is your transposed DataFrame
plot_gdp_bar_chart(df_transposed_GDP, 'GDP')

def plot_dot_plot_from_dict(data, ylabel, title):
    """
    Plots a dot plot for the given data dictionary.

    Parameters:
    - data (dict): The dictionary containing data in the specified format.
    - ylabel (str): The label for the y-axis.
    - title (str): The title of the plot.

    Returns:
    - None
    """
    # Create a DataFrame
    df = pd.DataFrame(data)
    df.set_index('Country Name', inplace=True)

    # Convert years to strings
    df.columns = df.columns.astype(str)

    # Plot the data as a dot plot
    plt.figure(figsize=(10, 6))

    for country in df.index:
        plt.plot(df.columns, df.loc[country], linestyle="dashed", label=country)

    # Adding labels and legend
    plt.xlabel('Year', fontweight='bold', fontsize=12)
    plt.ylabel(ylabel, fontweight='bold', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, axis='y', linestyle='', alpha=0.7)

    plt.show()

data = {
    'Country Name': ['China', 'Pakistan', 'United States'],
    '1990 [YR1990]': [13.186696, 38.592258, 20.272607],
    '2000 [YR2000]': [12.679720, 40.265670, 19.140966],
    '2015 [YR2015]': [12.235643, 39.435450, 17.124512],
    '2020 [YR2020]': [11.606259, 40.122976, 17.243857]
}

# Assuming data is your dictionary
plot_dot_plot_from_dict(data, 'Arable Land (% of land area)', 'Arable Land Over the Years')

def plot_dot_plot_from_dataframe(df, ylabel, title):
    """
    Plots a dot plot for the given DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing data for the dot plot.
    - ylabel (str): The label for the y-axis.
    - title (str): The title of the plot.

    Returns:
    - None
    """
    # Convert years to strings
    df.columns = df.columns.astype(str)

    # Plot the data as a dot plot
    plt.figure(figsize=(10, 6))

    for country in df.index:
        plt.plot(df.columns, df.loc[country], linestyle="dashed", label=country)

    # Adding labels and legend
    plt.xlabel(ylabel, fontweight='bold', fontsize=12)
    plt.ylabel('Year', fontweight='bold', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=14)
    plt.legend()
    plt.grid(True, axis='y', linestyle='', alpha=0.7)

    plt.show()

# Example usage:# Your data
data = {
    'Country Name': ['China', 'Pakistan', 'United States'],
    '1990 [YR1990]': [16.738007, 6.468958, 33.022308],
    '2000 [YR2000]': [18.853473, 5.852091, 33.130174],
    '2015 [YR2015]': [22.399821, 5.101443, 33.899723],
    '2020 [YR2020]': [23.431323, 4.833307, 33.866926]
}

# Create a DataFrame
df = pd.DataFrame(data)
df.set_index('Country Name', inplace=True)

# Convert years to strings
df.columns = df.columns.astype(str)
# Assuming df_transposed_forest_land is your DataFrame
plot_dot_plot_from_dataframe(df, 'Forest Land (% of land area)', 'Forest Land Over the Years')

def plot_country_correlation(country_name: str, file_names: List[str]) -> None:
    """
    Plots a correlation heatmap for a specified country using data from multiple files.

    Parameters:
    - country_name (str): The name of the country for which correlation is to be analyzed.
    - file_names (List[str]): A list of file names containing data for the analysis.

    Returns:
    - None
    """
    # Read the data from each file into a dictionary of DataFrames
    data = {}
    for file_name in file_names:
        data[file_name.split('.')[0]] = pd.read_csv(file_name, encoding='latin-1')

    # Extract data for the specified country from each file and set the file name as a column
    country_data = {key: df[df['Country Name'] == country_name]
                    .dropna(subset=['Country Name'])
                    .set_index('Country Name')
                    .transpose()
                    .rename(columns={0: key}) for key, df in data.items()}

    # Merge the DataFrames into a single DataFrame
    merged_country_data = pd.concat(country_data.values(), axis=1)

    # Keep only the years 1990, 2000, 2015, and 2020
    merged_country_data = merged_country_data.loc[['1990 [YR1990]', '2000 [YR2000]', '2015 [YR2015]', '2020 [YR2020]']]

    # Set column names as file names
    merged_country_data.columns = file_names

    # Display the modified country data
    print(merged_country_data)

    # Creating the correlation matrix of the country dataset
    country_corr_matrix = merged_country_data.corr()

    # Create the heatmap using the `heatmap` function of Seaborn
    sns.heatmap(country_corr_matrix, cmap='coolwarm', annot=True)
    plt.title(country_name)

    # Display the heatmap using the `show` method of the `pyplot` module from matplotlib.
    plt.show()

# Example usage:
country_name = 'China'
file_names = ['forest_land.csv', 'arable_land.csv', 'GDP.csv', 'CO2.csv', 'population.csv', 'energy_consumption.csv']
plot_country_correlation(country_name, file_names)

country_name = 'Pakistan'
file_names = ['forest_land.csv', 'arable_land.csv', 'GDP.csv', 'CO2.csv', 'population.csv', 'energy_consumption.csv']
plot_country_correlation(country_name, file_names)
