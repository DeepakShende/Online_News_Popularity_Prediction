# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

## Helper functions
def popularity(msg):
    return "This will be Popular" if msg else "Needs improvement"

def wordcount(msg):
    return len(msg.split())

def unique_wordcount(msg):
    return len(set(msg.split()))
  
### Flask methods
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    columns = ['title','article','channel','links','images','videos','day']
    
    df = pd.DataFrame(final_features,columns=columns)

    df['n_tokens_title'] = df['title'].apply(wordcount)
    #df['n_tokens_title'] = 14
    df['n_tokens_content'] = df['article'].apply(wordcount)
    #df['n_tokens_content'] = 493
    df['n_unique_tokens'] = df['article'].apply(unique_wordcount)
    
    for col in ['channel','links','images','videos','day']:
        df[col] = df[col].astype(float)
    
    
    df.drop(columns=['title','article'],inplace=True)
    order_columns = ['n_tokens_title','n_tokens_content','n_unique_tokens',
                     'links','images','videos','day']

    df = df.loc[:,order_columns]

    prediction = model.predict(df)

    output = prediction[0]

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)