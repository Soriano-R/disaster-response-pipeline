import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'omw-1.4'], quiet=True)
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

import pickle


def load_data(database_filepath):
    """
    Load data from SQLite database
    
    Parameters:
    database_filepath (str): Path to SQLite database
    
    Returns:
    X (pandas.Series): Feature data (messages)
    Y (pandas.DataFrame): Target data (categories)
    category_names (list): List of category names
    """
    # Load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    
    # Define features and target
    X = df['message']
    Y = df.iloc[:, 4:]  # All columns from the 5th column onwards
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    Process text data: tokenize, lemmatize, and clean
    
    Parameters:
    text (str): Text to be processed
    
    Returns:
    clean_tokens (list): List of cleaned tokens
    """
    # Replace URLs with a placeholder
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Tokenize, lemmatize, and clean
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline with GridSearchCV
    
    Returns:
    cv (GridSearchCV): GridSearchCV object with pipeline and parameter grid
    """
    # Create pipeline
    pipeline = Pipeline([
        ('cvect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
    ])
    
    # Define parameters for GridSearchCV
    parameters = {
        'cvect__max_features': [100, 200],
        'cvect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [50, 100]
    }
    
    # Create GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance
    
    Parameters:
    model: Trained model
    X_test (pandas.Series): Test features
    Y_test (pandas.DataFrame): Test targets
    category_names (list): List of category names
    
    Returns:
    None
    """
    # Predict on test data
    Y_pred = model.predict(X_test)
    
    # Print classification report for each category
    print("\nModel Performance:")
    print("="*80)
    
    # Overall accuracy
    accuracies = {}
    for idx, category in enumerate(category_names):
        accuracies[category] = accuracy_score(Y_test.iloc[:, idx], Y_pred[:, idx])
    
    # Print average accuracy
    avg_accuracy = np.mean(list(accuracies.values()))
    print(f"Average Accuracy: {avg_accuracy:.3f}")
    
    # Detailed evaluation for each category
    for idx, category in enumerate(category_names):
        print(f"\nCategory: {category}")
        print(classification_report(
            Y_test.iloc[:, idx], 
            Y_pred[:, idx],
            zero_division=0
        ))


def save_model(model, model_filepath):
    """
    Save trained model as a pickle file
    
    Parameters:
    model: Trained model to save
    model_filepath (str): Path to save the model
    
    Returns:
    None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Main function to execute the ML pipeline
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
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