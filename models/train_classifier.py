import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    '''
    Function to load data from a specified sqlite database
    -- Input --
        database_filepah: Location path to sqlite database
    -- Output --
        X: (pd.Series) Input variable for classifier. Contains text for messages
        Y: (pd.DataFrame) Output variables for classifier. Contains 36 categories 
        category_names: Column names for the 36 categories in Y. Needed for evaluate_model function
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disasterdata', engine)
    df_clean = df[np.isfinite(df['related'])]
    X = df_clean['message']  #['id','message','original','genre']
    Y = df_clean.drop(['id','message','original','genre'], axis = 1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    Function to be used in place of the built-in tokenizer in CountVectorize
    -- Input --
        text: Contains raw textfor each message
    -- Output --
        clean_words: Tokenized/lematized version of input text
    '''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]"," ", text)
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    clean_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
                           
    return clean_words


def build_model():
    '''
    Function to build the model pipeline and create a grid search object
    -- Input --
        N/A
    -- Output --
        cv: Grid Search Object that contains our created pipeline
    '''
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))    
    ])
    
    
    # specify parameters for grid search
    parameters = {'vectorizer__max_df': [.8],
                  'clf__estimator__n_estimators': [100]
                 }
    
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate the model and return accuracy scores
    -- Input --
        model: any model pipeline or grid search object that has been built/fit using our custom pipeline
        X_test: Input variables to test how the classifier performs
        Y_test: Output variables to compare how the classifier performs on the test input
        category_names: Names of the 36 categories in Y. Used to print out accuracy results for each category
    -- Output --
        N/A (although it does print out accuracy results, nothing is returned by this function)
    '''
    y_pred = model.predict(X_test)
    df_ypred = pd.DataFrame(y_pred, columns = Y_test.columns)
    df_ypred = df_ypred.reset_index(drop=True)
    y_test = Y_test.reset_index(drop=True)
    accuracy = (df_ypred == y_test).mean().mean()

    for col in y_test.columns:
        print(col)
        print(classification_report(y_test[col].values, df_ypred[col]))
        
    print("Accuracy:", accuracy)
    

def save_model(model, model_filepath):
    '''
    Function to save the model as a pickle file. This can eventually be used by our web app
    -- Input --
        model: any model pipeline or grid search object that has been built/fit using our custom pipeline
        model_filepath: Where you want the pkl file to be saved
    -- Output --
        cv: Grid Search Object that contains our created pipeline
    '''
    file_name = model_filepath
    pickle.dump(model, open(file_name, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
