import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function to load the data from a csv file and store it as a DataFrame
    
    -- Input --
        messages_filepath -> Location where the messages csv file lives
        categories_filepath -> Location where the categories csv file lives
    -- Output --
        df: DataFrame containing the information from both the messages csv file and the categories csv file
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    """
    Function to clean the DataFrame containing all the data into a usable format
    
    -- Input --
        df: DataFrame containing input csv files
    -- Output --
        df: Cleaned version of input DataFrame
    """
    categories = df['categories']
    categories = categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    df = df.drop(['categories'], axis = 1)
    df = pd.concat([df,categories], axis = 1)
    df.drop_duplicates(keep='first',inplace=True)
    return df

def save_data(df, database_filename):
    """
    Function to save the DataFrame to an sqlite database for later use
    
    -- Input --
        df: Cleaned DataFrame object
        database_filename: Desired name of sqlite database
    -- Output --
        N/A
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disasterdata', engine, index=False) 


def main():
    """
    Main Function to Extract, Transform, and then Load the data to an sqlite database
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
