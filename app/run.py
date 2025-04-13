import json
import plotly
import pandas as pd
import numpy as np
import os

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as graph_obj
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# Define colorblind-friendly color palette
# Blue, Yellow, Green, Red, Purple, Orange, Light Blue
COLORBLIND_COLORS = [
    '#0072B2',  # Blue
    '#E69F00',  # Yellow/Orange
    '#009E73',  # Green
    '#CC79A7',  # Pink/Purple
    '#56B4E9',  # Light Blue
    '#D55E00',  # Red/Orange
    '#F0E442'   # Light Yellow
]

def tokenize(text):
    """
    Process text data: tokenize, lemmatize, and clean
    
    Parameters:
    text (str): Text to be processed
    
    Returns:
    clean_tokens (list): List of cleaned tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load data - Using the path you specified in your error log
database_filepath = '/Users/soriano/Documents/Macbook/Udacity/Data_Scientist/Data_Engineering/disaster_response_pipeline_project/data/DisasterResponse.db'
engine = create_engine(f'sqlite:///{database_filepath}')
try:
    df = pd.read_sql_table('DisasterResponse', engine)
    print(f"Successfully loaded database from {database_filepath}")
except Exception as e:
    print(f"Error loading database: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Database path tried: {database_filepath}")
    raise

# Load model - Update path similarly
model_filepath = '/Users/soriano/Documents/Macbook/Udacity/Data_Scientist/Data_Engineering/disaster_response_pipeline_project/models/classifier.pkl'
try:
    model = joblib.load(model_filepath)
    print(f"Successfully loaded model from {model_filepath}")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Model path tried: {model_filepath}")
    raise


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Render the home page with visualizations
    """
    # extract data needed for visuals
    # Count of messages by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Distribution of message categories - 
    category_names = df.iloc[:, 4:].columns  
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    if 'message_len' in category_counts.index:
        category_counts = category_counts.drop('message_len')
    # Calculate category ratios (proportion of messages with each category)
    total_messages = len(df)
    category_ratios = (category_counts / total_messages).round(2)
    
    # Distribution of message length
    df['message_len'] = df['message'].apply(len)
    message_lengths = df.groupby('genre')['message_len'].mean()
    
    # create visuals
    graphs = [
        # GRAPH 1 - Distribution by Genre
        {
            'data': [
                graph_obj.Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color=COLORBLIND_COLORS[0])
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'plot_bgcolor': 'rgba(245, 245, 245, 1)',
                'paper_bgcolor': 'rgba(245, 245, 245, 1)'
            }
        },
        
        # GRAPH 2 - All Categories Distribution as ratio (percentage)
        {
            'data': [
                graph_obj.Bar(
                    x=category_ratios.index,
                    y=category_ratios.values,
                    marker=dict(
                        color=COLORBLIND_COLORS[1],
                        line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
                    )
                )
            ],

            'layout': {
                'title': 'Message Categories (% of All Messages)',
                'yaxis': {
                    'title': "Percentage (%)"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45
                },
                'height': 700,
                'margin': {
                    'b': 150,
                    'l': 80,
                    'r': 80
                },
                'plot_bgcolor': 'rgba(245, 245, 245, 1)',
                'paper_bgcolor': 'rgba(245, 245, 245, 1)'
            }
        },
        
        # GRAPH 3 - Average Message Length by Genre
        {
            'data': [
                graph_obj.Bar(
                    x=genre_names,
                    y=message_lengths,
                    marker=dict(
                        color=COLORBLIND_COLORS[2],
                        line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
                    )
                )
            ],

            'layout': {
                'title': 'Average Message Length by Genre',
                'yaxis': {
                    'title': "Average Length (characters)"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'plot_bgcolor': 'rgba(245, 245, 245, 1)',
                'paper_bgcolor': 'rgba(245, 245, 245, 1)'
            }
        },
        
        # GRAPH 4 - Heatmap of category co-occurrence 
        # (replaces pie chart with something more useful)
        {
            'data': [
                graph_obj.Heatmap(
                    z=df.iloc[:, 4:11].corr().values,  # Just using first 7 columns for clarity
                    x=list(df.iloc[:, 4:11].columns),
                    y=list(df.iloc[:, 4:11].columns),
                    colorscale='YlGnBu',
                    zmin=-1,
                    zmax=1
                )
            ],

            'layout': {
                'title': 'Correlation Between Top Categories',
                'height': 500,
                'margin': {
                    'l': 150,
                    'r': 20,
                    'b': 150,
                },
                'paper_bgcolor': 'rgba(245, 245, 245, 1)'
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Handle the classification request and show results
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()