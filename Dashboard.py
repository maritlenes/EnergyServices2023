import dash
from jupyter_dash import JupyterDash
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from dash import dcc
from dash import html
from dash import Dash
import pandas as pd

import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# data and data2019 are the cleaned and feature selected
# data, respectively for 2017-2018 and 2019.  
data = pd.read_csv('df_final.csv')
data2019 = pd.read_csv('data2019.csv')

# Some additional cleaning and prepearing 
data['Date'] = pd.to_datetime(data['Date'])
data2019['Date'] = pd.to_datetime(data2019['Date'])

data=data.set_index(['Date'],drop=True)
data2019=data2019.set_index(['Date'],drop=True)

data.rename(columns = {'Power [kW]': 'Power [kWh]'}, inplace = True)
data.rename(columns = {'Power-1 [kW]': 'Power-1 [kWh]'}, inplace = True)

# Prepearing the line plots of the real measured energy consumption
data_features=data.iloc[:,1:5]
data_X=data_features.values
data_Y = data.values[:,0]

fig2017 = px.line(data.loc['2017'], x=data.loc['2017'].index, y=data.loc['2017']['Power [kWh]']).update_layout(
    xaxis_title = 'Date',yaxis_title = 'Power [kWh]')

fig2018 = px.line(data.loc['2018'], x=data.loc['2018'].index, y=data.loc['2018']['Power [kWh]']).update_layout(
    xaxis_title = 'Date',yaxis_title = 'Power [kWh]')

data_features2019=data2019.iloc[:,1:5]
data_X2019=data_features2019.values
data_Y2019=data2019.values[:,0]

fig2019 = px.line(data2019.loc['2019'], x=data2019.loc['2019'].index, y=data2019.loc['2019']['Power [kWh]']).update_layout(
    xaxis_title = 'Date',yaxis_title = 'Power [kWh]')

#Load saved RF model from project 1
with open('RF_model.pkl','rb') as file:
    RF_model=pickle.load(file)

y_pred_RF = RF_model.predict(data_X)
y_pred_RF2019 = RF_model.predict(data_X2019)

# Prepearing the forcast line plots of the model generated energy consumption
df_results = data.copy()
df_results['Random Forest Model'] = y_pred_RF
df_results['Date'] = df_results.index


df_results2019 = data2019.copy()
df_results2019['Random Forest Model'] = y_pred_RF2019
df_results2019['Date'] = df_results2019.index


new_lable_names = {'Power [kWh]': "Real measured power [kWh]", 'Random Forest Model': "Model generated power [kWh]"}

figf2017 = px.line(df_results.loc['2017'],x='Date',y=['Power [kWh]','Random Forest Model']).update_layout(
yaxis_title = 'Power [kWh]')
figf2017.for_each_trace(lambda t: t.update(name = new_lable_names[t.name]))

figf2018 = px.line(df_results.loc['2018'],x='Date',y=['Power [kWh]','Random Forest Model']).update_layout(
yaxis_title = 'Power [kWh]')
figf2018.for_each_trace(lambda t: t.update(name = new_lable_names[t.name]))

figf2019 = px.line(df_results2019.loc['2019'],x='Date',y=['Power [kWh]','Random Forest Model']).update_layout(
yaxis_title = 'Power [kWh]')
figf2019.for_each_trace(lambda t: t.update(name = new_lable_names[t.name]))

# Auxiliary function for generating a table for the errors
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


# Auxiliary function to generate an error matrix
def generate_error_matrix(y_real, y_pred):
    MAE=metrics.mean_absolute_error(y_real,y_pred) 
    MBE=np.mean(y_real-y_pred)
    MSE=metrics.mean_squared_error(y_real,y_pred)  
    RMSE= np.sqrt(metrics.mean_squared_error(y_real,y_pred))
    cvRMSE=RMSE/np.mean(y_real)
    NMBE=MBE/np.mean(y_real)
    
    err = [MAE,MBE,MSE,RMSE,cvRMSE,NMBE]
    error = {'Method': ['Random Forest'], 'MAE': [err[0]],'MBE' : [err[1]], 'MSE': [err[2]], 'RMSE': [err[3]],'cvMSE': [err[4]], 'NMBE' : [err[5]]}
    error_metrix = pd.DataFrame(error)
    return error_metrix

#Evaluating the errors
#First: split the 2017 and 2018 data, to make it easier
data_features2017=data.loc['2017'].iloc[:,1:5]
data_X2017=data_features2017.values
data_Y2017 = data.loc['2017'].values[:,0]

data_features2018=data.loc['2018'].iloc[:,1:5]
data_X2018=data_features2018.values
data_Y2018 = data.loc['2018'].values[:,0]

error_metrics2017 = generate_error_matrix(data_Y2017,df_results.loc['2017']['Random Forest Model'])
error_metrics2018 = generate_error_matrix(data_Y2018,df_results.loc['2018']['Random Forest Model'])
error_metrics2019 = generate_error_matrix(data_Y2019,y_pred_RF2019)

# Create the app 
app = JupyterDash(__name__, external_stylesheets=external_stylesheets) # create the dash app
server = app.server

app.layout = html.Div([
    html.H2('IST South Tower Electricity Consumption',style={"font-weight": "bold"}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Electricity Consumption 2017-2019', value='tab-1'),
        dcc.Tab(label='Electricity Model Consumption 2017-2019', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('Electricity consumption (kWh) 2017',style={"font-weight": "bold"}),
            dcc.Graph(
                id='yearly-data',
                figure=fig2017,
            )
             ]),html.Div([
            html.H4('Electricity consumption (kWh) 2018',style={"font-weight": "bold"}),
            dcc.Graph(
                id='yearly-data',
                figure=fig2018,
                ) 
        ]),html.Div([
            html.H4('Electricity consumption (kWh) 2019',style={"font-weight": "bold"}),
            dcc.Graph(
                id='yearly-data',
                figure=fig2019,
                ) 
        ]) 
            
    elif tab == 'tab-2':
        return html.Div([
            html.H4('Electricity Model Consumption (kWh) 2017',style={"font-weight": "bold"}),
            dcc.Graph(
                id='yearly-data',
                figure=figf2017,
                ),
            html.H6('Model error matrix 2017:'),
            generate_table(error_metrics2017)
        ]),html.Div([
            html.H4('Electricity Model Consumption (kWh) 2018',style={"font-weight": "bold"}),
            dcc.Graph(
                id='yearly-data',
                figure=figf2018,
                ),
            html.H6('Model error matrix 2018:'),
            generate_table(error_metrics2018)
        ]),html.Div([
            html.H4('Electricity Model Consumption (kWh) 2019',style={"font-weight": "bold"}),
            dcc.Graph(
                id='yearly-data',
                figure=figf2019,
                ),
            html.H6('Model Error Matrix 2019:'),
            generate_table(error_metrics2019)
        ])

# Running the app 
if __name__ == '__main__':
    app.run_server()
