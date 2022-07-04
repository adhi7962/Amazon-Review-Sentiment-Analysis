from optparse import Values
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import numpy as np
from numpy import positive 
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from sklearn.feature_extraction.text import CountVectorizer
import nltk
# Remove stopwords
from nltk.corpus import stopwords
#nltk.download('punkt')
# Tokenization
from nltk.tokenize import word_tokenize

# Graphs
def drawpiechart(fig=None):
    if(fig is None):
        fig=asd['pie_chart']
    return  html.Div([dbc.Card(dbc.CardBody([html.P('Ratings Visualization'),
    dcc.Graph(figure=fig.update_layout(template='plotly_dark',plot_bgcolor= 'rgba(0, 0, 0, 0)',
    paper_bgcolor= 'rgba(0, 0, 0, 0)',), 
    config={'displayModeBar': False},)])), ])

class reviewanalysis:
    def get_analysis(data,id):
        input_data = data[data['overall'].apply(lambda x: x ==id)]#.sort_values(by ='Rating',ascending=False)
        business_name = input_data.iloc[0]['overall']
        business_rating=input_data.iloc[0]['Rating']
        num_review=len(data)
        counts = data.overall.value_counts()
        sentiment_fig = px.pie(counts, values='overall')
        output = {}
        
        output['business_name']=business_name
        output['num_review']=num_review
        output['pie_chart']= sentiment_fig 
        output['overall rating']= business_rating
        return output 

 

# text
def get_recent_reviews(data,id):
    data = data[data['overall'].apply(lambda x: x ==id)]#.sort_values(by ='Rating',ascending=False)
    data=data.drop(columns=['Title','cleaned_reviews'])
    return data

def drawBusinessName(): 
    return html.Div([
    html.H2('Customer Review Analysis',style={'height':'62px','width':'390px','padding':'1px'}),    
    dcc.Dropdown(id='business_name',
    options=[
    {'label': 'Good review', 'value': 'positive'},
    {'label': 'Bad review', 'value': 'negative'},
    {'label': 'Neutral review', 'value': 'neutral'},
    ],
    placeholder="Select Rating",)])  

def drawText(text = None):
    if text is None:
        text = asd['business_name']        
    return html.Div([
    dbc.Card(
    dbc.CardBody([html.Div([html.P('Selected Business',style={'color': 'white'}),html.H5(text,style={'color': 'yellow'}), ], style={'textAlign': 'center'}) 
    ])),])
    
def drawTextOverallRating(text = None):
    if text is None:
        text = asd['overall rating']        
    return html.Div([
    dbc.Card(dbc.CardBody([html.Div([html.P('Overall Rating',style={'color': 'white'}),html.H5(text,style={'color': 'yellow'}), ], style={'textAlign': 'center'}) 
    ])),])

def drawTotalReview(text = None):
    if text is None:
        text = asd['num_review']
    return html.Div([
        dbc.Card(
        dbc.CardBody([html.Div([html.P('Total Review',style={'color': 'white'}), 
        html.H5(text,style={'color': 'yellow'}), 
        ], style={'textAlign': 'center'}) ])),])


def drawRecentReviews(data = None):
    if data is None:
        data = recent_reviews
    return  html.Div([
    dbc.Card(
    dbc.CardBody([html.P('Recent Reviews'),
    dbc.Table.from_dataframe(data, striped=False, bordered=False, hover=True,dark=True,responsive=True)
    ])),], style={'textAlign': 'center',"maxHeight": "500px", "overflow": "scroll"})


          
    # fig = asd[pir_chart]
    # #plot_bgcolor=colors['background'],
    # paper_bgcolor=colors['background'],
    # font_color=colors['text']
    # )
    # return dcc.Graph(
    #     id='example-graph',
    #     figure= fig
    # )

app = Dash(__name__,external_stylesheets=[dbc.themes.SLATE])
data = pd.read_csv(r'C:\Users\Hxtreme\cleaned_review.csv',index_col = 0)
cr = reviewanalysis
asd = cr.get_analysis(data,'positive')
recent_reviews=get_recent_reviews(data,'positive') 
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


app.layout = html.Div([
    dbc.Card(
    dbc.CardBody([
    dbc.Row([dbc.Col([html.H1('Amazon Review'),html.H3('Here, this dashboard shows overall view of product.')])],align='center'),  
    dbc.Row([dbc.Col([drawBusinessName()], width=3), dbc.Col([drawText()], width=3,id='sb01'),dbc.Col([drawTextOverallRating()], width=3,id='sb02'),dbc.Col([drawTotalReview()], width=3,id='sb03')],align='center'),
    dbc.Row([dbc.Col([drawpiechart ()], width=5,id='df02'),dbc.Col([drawRecentReviews(recent_reviews)], width=7,id='df04')],align='center'),
    ]), color = 'dark')
  ])


@app.callback(
    [Output(component_id='sb01', component_property='children'),
    Output(component_id='sb02', component_property='children'),
    Output(component_id='sb03', component_property='children'),
    Output(component_id='df02', component_property='children'),
    Output(component_id='df04', component_property='children')],
    [Input(component_id='business_name', component_property='value')]
)
def update_output_div(input_value):
    cr = reviewanalysis       
    asd = cr.get_analysis(data,input_value)
    recent_reviews = get_recent_reviews(data,input_value)  
    return drawText(asd['business_name']),drawTextOverallRating(asd['overall rating']),drawTotalReview(asd['num_review']),drawRecentReviews(recent_reviews),drawpiechart(asd['pie_chart'])


if __name__ == '__main__':
    app.run_server(debug=True)
