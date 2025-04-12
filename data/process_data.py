import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from message and categories CSV files
    
    Parameters:
    messages_filepath (str): Path to messages CSV file
    categories_filepath (str): Path to categories CSV file
    
    Returns:
    df (pandas.DataFrame): Merged dataframe containing messages and categories
    """
    # Load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge the datasets
    df = messages.merge(categories, on='id', how='inner')
    
    return df


def clean_data(df):
    """
    Clean the merged dataframe by splitting categories, converting to binary values,
    and removing duplicates
    
    Parameters:
    df (pandas.DataFrame): Merged dataframe to clean
    
    Returns:
    df (pandas.DataFrame): Cleaned dataframe
    """
    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract column names from the first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    
    # Convert category values to just 0 or 1
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        # Ensure binary values (0 or 1) - replace anything > 1 with 1
        categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)
    
    # Drop the original categories column
    df = df.drop('categories', axis=1)
    
    # Concatenate the original df with the new categories df
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Save the cleaned dataframe to an SQLite database
    
    Parameters:
    df (pandas.DataFrame): Cleaned dataframe to save
    database_filename (str): Filename for the SQLite database
    
    Returns:
    None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    """
    Main function to execute the ETL pipeline
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()