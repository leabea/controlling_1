# Diese Python-Datei 'webanwendung.py' erstellt die Webanwendung 'Forecasting im Controlling' zur Prognose
#von Zeitreihen mit jährlichem Zeitabstand.

#Kommentare (wie dieser) beginnen mit '#' und sind grün hervorgehoben.

#Zunächst erfolgt das Importieren aller relevanter Module und Bibliotheken --> Erklärung siehe Anhang 1 der Masterthesis
import dash
import dash_core_components as dcc
from dash import html
from dash import dcc
import numpy as np
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State
import io
import base64
import dash_bootstrap_components as dbc
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go
import pmdarima as pm
from pmdarima.arima import auto_arima

#App erstellen, wobei ich hier auf das Design 'dbc.themes.COSMO' zurückgreife
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO], suppress_callback_exceptions=True)
server = app.server

#layout ist ein HTML-Element, das den graphischen Aufbau der Seite festlegt.
layout = html.Div([
    #Definition, wie der Header aussehen soll
    html.H1(children=['Forecasting im Controlling', 
            html.H2('Jährliche Prognosen', style={'fontSize': '16px','marginTop': '-15px'})],
            style={
                'textAlign': 'center',
                'width': '100%',
                'height': '100px',
                'backgroundColor': '#0B2896',
                'lineHeight': '90px',
                'color': '#ffffff'
                }),
    html.Div(style={'height': '50px'}),
    html.Div([
        html.Div(
            #an dieser Stelle wird der Button 'CSV-Datei hochladen' spezifiziert
            dcc.Upload(
                id='upload-data',
                children=html.Div([html.A('CSV-Datei hochladen')]),
                style={
                    'width': '20%',
                    'height': '80px',
                    'lineHeight': '70px',
                    'backgroundColor': '#0B2896',
                    'borderWidth': '5px',
                    'borderStyle': 'solid' ,
                    'borderColor': 'grey',
                    'borderRadius': '30px',
                    'textAlign': 'center',
                    'margin': '0 auto',
                    'color': '#ffffff',
                    'font-size': '30px'
                },
                multiple=False
            ),
            style={ 'textAlign': 'center', 'width': '100%'}
        ),
        #Unterhalb des Buttons soll ein Hinweistext auf der Seite zu sehen sein
        html.Div(html.P(children=['Mit einem Klick auf den Button, können Sie eine CSV-Datei auswählen. Diese sollte aus zwei Spalten bestehen. Die erste Spalte sollte Datumsangaben (chronologisch sortiert) beinhalten.',html.Br(),'Die zweite Spalte besteht aus den zugehörigen, numerischen Werten. Sie können die hochgeladene Datei jederzeit über den Button ändern oder Ihre Eingabe rückgängig machen, indem Sie die Seite neu laden.'], style={'font-size': '17px', 'textAlign': 'center', 'margin': '20px' }
    )),
        html.Div(html.P(children=['Anzahl der Perioden, die prognostiziert werden sollen:',html.Br()], style={'font-size': '20px', 'textAlign': 'center', 'margin': '20px', 'color': '#0B2896'}
    ))
        ]),
            html.Div(
        style={
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center'
        },
        children=[        
            dcc.Dropdown(
                #Dropdown-Menü, das festlegt, wie viele zusätzliche Perioden des Datensatzes vorhergesagt werden sollen. Per Default ist eine Periode eingestellt.
                id='periods-dropdown',            
                options=[{'label': '1', 'value': '1'},
                        {'label': '2', 'value': '2'}],
                    value='1',
                    style={
                        'width': '40%',
                        'height': '50px',
                        'font-size': '20px',
                        'borderWidth': '3px',
            
                        'textAlign': 'center',
                        'margin': '0 auto'
                    }
            )
        ]
    ),
    html.Div(style={'height': '50px'}),
    html.Div([
        html.Div(style={'width': '5%', 'height': '900px'}),
        html.Div([
            dcc.Dropdown(
                #Dropdown-Menü mit herkömmlichen Prognosemethoden. Per Default ist die lineare Regression eingestellt.
                id='dropdown1',
                options=[
                    {'label': 'lineare Regression', 'value': 'linReg'},
                    {'label': 'Moving Average', 'value': 'movingAverage'},
                    {'label': 'Exponential Smoothing', 'value': 'exponentialSmoothing'}
           
                ],
                #wenn der Nutzer keine andere Forecasting-Methode auswählt, soll die klassiche lineare Regression durchgeführt werden
                value = 'linReg',
                style={'font-size': '20px', 'borderWidth': '3px', 'margin': '10px'}
            ),
           
            html.Div([
                #Unterhalb des Dropdowns soll das Liniendiagramm angezeigt werden.
                html.Div(id='output-data-upload-links', style={'width': '100%', 'display': 'inline-block'})
            ]),
            html.Div(id='textContainerLinks', style={'font-size': '20px'})
            #an dieser Stelle soll der Text unterhalb des Diagramms angegeben werden
        ], className='col', style={'float': 'left', 'width': '42.5%', 'backgroundColor': '#ffffff', 'borderStyle': 'solid' ,
                    'borderColor': '#ffffff',
                    'borderRadius': '15px','borderWidth': '5px', 'height': '1500px'}),
        html.Div(style={'width': '5%', 'height': '900px'}),
        html.Div([
            dcc.Dropdown(
                #Dropdown-Menü mit den Prognosemethoden, die auf Machine Learning Basieren. Per Default ist die lineare Regression eingestellt.
                id='dropdown2',
                options=[
                    {'label': 'lineare Regression mit Gewichtungsfunktion', 'value': 'linearRegression'},
                    {'label': 'Auto-ARIMA', 'value': 'autoArima'},
                    {'label': 'Ridge Cross Validation', 'value': 'ridgeCV'}
                ],
                #Trifft der Nutzer keine Auswahl im Block rechts auf der Seite, wird die gewichtete lineare Regression aufgerufen
                value = 'linearRegression',
                style={'font-size': '20px', 'borderWidth': '3px', 'margin': '10px'}
                
            ),
            html.Div([
                #Unterhalb des Dropdowns soll das Liniendiagramm erscheinen
                html.Div(id='output-data-upload', style={'width': '100%', 'display': 'inline-block'})
            ]),
            html.Div(id='textContainerRechts', style={'font-size': '20px'})
            #an dieser Stelle soll der Text unterhalb des Diagramms angegeben werden
        ], className='col', style={'float': 'right', 'width': '42.5%', 'backgroundColor': '#ffffff', 'borderStyle': 'solid' ,
                    'borderColor': '#ffffff',
                    'borderRadius': '15px','borderWidth': '5px', 'height': '1500px'}),
        html.Div(style={'width': '5%', 'height': '900px'})
    ], className='row'), 
    
], style={'backgroundColor': '#F0F5F7', 'height': '3000px'})

#lesen der CSV-Datei und Weiterverarbeitung als Pandas-Dataframe (tabellarisch)
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        df = pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
        return df
    else:
        return None

######## Liste mit den Mean Absolute Errors oder Abweichungsarrays ######

#Der MAE wird auf der Seite 'Forecasting im Controlling' nur für Machine Learning Modelle ausgegeben 
mae_rechts = [0, 0, 0]

#Arrays, die zunächst nur mit Nullen gefüllt werden, um später geändert zu werden. 
# Dann soll darin die Abweichung des letzten (realen) Datenpunkts der Zeitreihe mit dem dafür prognostizierten Wert verglichen werden (durch einfache Differenzbildung im Betrag).
abweichungen_links = [0, 0, 0]
abweichung_rechts = [0,0,0]

#Arrays, die zunächst nur mit Nullen gefüllt werden, um später geändert zu werden. 
# Dann soll darin die prozentuale Abweichung des letzten (realen) Datenpunkts der Zeitreihe mit dem dafür prognostizierten Wert prozentual verglichen werden.
proz_abweichung_links = [0,0,0]
proz_abweichung_rechts = [0,0,0]
    
###### MA 3 #######################################

def moving_average(data, ordnung):
    #Quelle: https://bougui505.github.io/2016/04/05/moving_average_in_python.html
    #Berechnet den gleitenden Durchschnitt für die gegebene Datenreihe mit der gegebenen Ordnung (hier k=3).
    window = np.ones(int(ordnung)) / float(ordnung)
    return np.convolve(data, window, 'same')

############linke Seite#############################  

@app.callback(
    #die Callback-Funktion spezifiziert notwendige Inputs, die die untenstehende Funktion update_output2 aktivieren, sowie was als Output ausgegeben wird. Darüberhinaus spzifiziert State den Zustand. Ändert sich dieser, wird die Funktion erneut ausgeführt.
    Output('output-data-upload-links', 'children'),
    Input('dropdown1', 'value'),
    Input('periods-dropdown', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
#Funktion 'update_output2' zur Durchführung der Prognosen und Berechnung der Abweichungen 
#hier dient als Input die eingelesene CSV-Datei, der Wert aus dem linken Dropdown Menü (zur Auswahl einer traditionellen Forecasting-Methode) und die Anzahl der zu prognostizierenden Perioden. Trifft der Nutzer keine Auswahl, erfolgt nach korrekt eingelesener CSV-Datei die Berechnung der nächsten Periode mithilfe der klassischen linearen Regression.
def update_output2(selected_graph, periods, list_of_contents, list_of_names):
    if selected_graph == 'linReg'  and periods == '1':
        #Eingaben müssen befüllt sein, sonst kommt es zu Fehlern
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            #Zusammenführung der Inhalte in einem Pandas Dataframe (Tabelle)
            df = parse_contents(contents, filename)
            if df is None:
                #bei leerer oder ungültiger CSV-Datei, die nun als Dataframe abgespeichert ist, wird eine Fehlermeldung an den Nutzer zurückgegeben.
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
            #alle Nan-Werte (ungültige Werte) aus dem Dataframe (Tabelle mit den Daten aus der CSV-Datei) löschen    
            df = df.dropna()
            #letzten Index der Tabelle in einer Variable speichern, um die Tabelle um einen Index weiter zu verlängern (das heißt eine Reihe wird hinzugefügt zu df)
            last_index = df.iloc[-1, 0]
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            
            #der Datensatz darf nicht leer sein
            if df is not None:
                #leere Zeilen des Datensatzes löschen, damit die Regression durchgeführt werden kann
                df2 = df.dropna()
                #X sei die Spalte mit den Datumsangaben und Y Spalte mit den zugehörigen Werten
                x = df2.iloc[:, [0]].values.flatten()
                y = df2.iloc[:, [1]].values.flatten()
                
                #Durchführung der linearen Regression
                #Quelle:Scipy (o.D. b): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
                beta, alpha, r_value, p_value, std_err = linregress(x, y)
                nextPrediction =    beta * (last_index + 1) + alpha

                #Hinzufügen der Vorhersage zum Dataframe 'df', damit die Zeitreihe inklusive der Prognose grahisch
                #dargestellt werden kann. 
                df.iloc[:, 1][df.index[-1]] = nextPrediction

                
                # Erstellung des Dataframe ('df3') als Kopie von 'df' allerdings nur bis zur vorletzten Reihe 
                df3 = df.iloc[:-1]
                df3 = df3.dropna()
                #Als x2-Werte werden die Zeitwerte und als y2 die dazugehörigen Werte herangezogen.
                x2 = df3.iloc[:, [0]].values.flatten()
                y2 = df3.iloc[:, [1]].values.flatten()
                
                #da hier eine Berechnung des MAE nicht möglich ist, nehme ich einen Datenpunkt weg,
                # sage ihn vorher und berechne auf Grundlage darauf den MAE
                beta2, alpha2, r_value, p_value, std_err = linregress(x2, y2)
                actual_value = y[-1]
                next_prediction2 =  beta2 * (last_index) + alpha2
                
                #Berechnung der Abweichungen im Absolutbetrag (kann einfach mit der Funktion
                # 'mean_absolute_error' berechnet werden, allerdigs entspricht das nicht dem MAE analog zu den Machine Learning Modellen, da hier nur ein Punkt betrachtet wird und nicht rund 20 % des Gesamtdatensatzes!!)
                mae = mean_absolute_error([actual_value], [next_prediction2])
                mae = mae.round(2)
                abweichungen_links[0] = mae
                #Berechnung der prozentualen Abweichung:
                proz_abweichung = (abs([next_prediction2][0] - [actual_value][0]) / [actual_value][0]) * 100
                proz_abweichung =  proz_abweichung.round(2)
                proz_abweichung_links[0] = proz_abweichung
                
                #Nun soll der Graph (visuell) definiert werden:
                fig = px.line(title='Lineare Regression (klassisch)')
                
                fig.add_trace(go.Scatter(
                    x=df.iloc[:-1, 0],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(
                    xaxis=dict(
                        title="Zeit",
                        tickmode="linear",
                        tick0=df.iloc[0, 0],
                        dtick=2,
                        range=[df.iloc[0, 0], df.iloc[-1, 0]],
                    ),
                    yaxis=dict(
                        title="Wert"
                    )
                )


                return dcc.Graph(
                    id='linregression-plot',
                    figure=fig
                )
  
        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })
    #falls der Nutzer 2 Perioden der Zukunft mithilfe der linearen Regression prognostizieren lassen möchte, wird dieser Teil der Funktion 'update_output2' durchgeführt.
    elif selected_graph == 'linReg' and periods == '2':
        #Eingaben müssen befüllt sein, sonst kommt es zu Fehlern
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            #Zusammenführung der Inhalte in einem Pandas Dataframe 'df' (in Tabellenform)
            df = parse_contents(contents, filename)
            #Ist dieser Dataframe nun leer, bekommt der Nutzer eine rote Fehlermeldung mit einer Pixelgröße von 20 Pixeln angezeigt.
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })            
            
            df = df.dropna()
            #letzten Index des Dataframes in einer Variable speichern, um das Dataframe anschließend um zwei Reihen zu erweitern mithilfe des letzten Index.
            last_index = df.iloc[-1, 0]
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            df = df.append({df.columns[0]: (last_index + 2)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                #X sei die Spalte mit den Datumsangaben und Y die zugehörigen Werte
                x = df2.iloc[:, [0]].values.flatten()
                y = df2.iloc[:, [1]].values.flatten()
                #Durchführung der linearen Regression mit dem vorgefertigten Modul linregress.
                #Quelle:Scipy (o.D. b): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
                beta, alpha, r_value, p_value, std_err = linregress(x, y)
                #Vorhersage des nächsten Werts:
                nextPrediction =    beta * (last_index + 1) + alpha
                #Vorhersage des übernächsten Werts:
                nextPrediction2 =   beta * (last_index + 2) + alpha

                #Hinzufügen der vorhergesagten Werte zu 'df'
                df.iloc[:, 1][df.index[-2]] = nextPrediction
                df.iloc[:, 1][df.index[-1]] = nextPrediction2          

                #neuer Datensatz 
                df3 = df.iloc[:-2]
                df3 = df3.dropna()
                x2 = df3.iloc[:, [0]].values.flatten()
                y2 = df3.iloc[:, [1]].values.flatten()
                
                #da hier eine Berechnung des MAE nicht möglich ist, nehme ich einen Datenpunkt weg, 
                # sage ihn vorher und berechne auf Grundlage darauf den MAE
                beta2, alpha2, r_value, p_value, std_err = linregress(x2, y2)
                actual_value = y[-1]
                next_prediction2 =  beta2 * (last_index) + alpha2
                
                #hier erfolgt keine Berechnung der Abweichung, weil die oben schon durchgeführt wurde.
                
                #Nun soll der Graph definiert werden:
                fig = px.line(title='Lineare Regression (klassisch)')
                
                fig.add_trace(go.Scatter(
                    x=df.iloc[:-2, 0],
                    y=df.iloc[:-2, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    x=[df.iloc[-3, 0],df.iloc[-2, 0], df.iloc[-1, 0]],
                    y=[df.iloc[-3, 1],df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(
                    xaxis=dict(
                        title="Zeit",
                        tickmode="linear",
                        tick0=df.iloc[0, 0],
                        dtick=2,
                        range=[df.iloc[0, 0], df.iloc[-1, 0]],
                    ),
                    yaxis=dict(
                        title="Wert"
                    )
                )
                return dcc.Graph(
                    id='linregression-plot1',
                    figure=fig
                )
    

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })
    #falls der Nutzer eine Periode der Zukunft mithilfe des Exponential Smoothings prognostizieren lassen möchte, wird dieser Teil der Funktion 'update_output2' durchgeführt.
    elif selected_graph == 'exponentialSmoothing' and periods == '1':
        #Eingaben müssen befüllt sein, sonst kommt es zu Fehlern
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            #Zusammenführung der Inhalte in einem Pandas Dataframe 'df' (in Tabellenform)
            df = parse_contents(contents, filename)
            #Ist dieser Dataframe nun leer, bekommt der Nutzer eine rote Fehlermeldung mit einer Pixelgröße von 20 Pixeln angezeigt.
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })            
            
            df = df.dropna()
            #Speicherung des letzten Index des Dataframes 'df', um danach den Dataframe (Tabelle) um eine Reihe zu erweitern 
            last_index = df.iloc[-1, 0]
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                #Definition der x- und y-Werte analog zur linearen Regression
                x = df2.iloc[:, [0]].values.flatten()
                y = df2.iloc[:, [1]].values.flatten()
                
                #Exponential Smoothing berechnen
                #Quelle: KoalaTea (2021): https://koalatea.io/python-ses-timeseries/
                alpha = 0.2
                smoothed_values = [y[0]]
                for i in range(1, len(y)):
                    smoothed_value = alpha * y[i] + (1 - alpha) * smoothed_values[i-1]
                    smoothed_values.append(smoothed_value)
                
                #Vorhersage des nächsten Index (der neue Index wurde bereits zum Dataframe hinzugefügt)
                nextPrediction = smoothed_values[-1]
                #Berechnung der Abweichungen im Absolutbetrag (kann einfach mit der Funktion 'mean_absolute_error' berechnet werden, allerdigs entspricht das nicht dem MAE analog zu den Machine Learning Modellen, da hier nur ein Punkt betrachtet wird und nicht rund 20 % des Gesamtdatensatzes!!)
                mae = mean_absolute_error(y[-1:], smoothed_values[-2:-1])
                mae = mae.round(2)
                abweichungen_links[1] = mae
                #Berechnung der prozentualen Abweichung
                proz_abweichung = (abs(smoothed_values[-2:-1][0] - y[-1:][0]) / y[-1:][0]) * 100
                proz_abweichung =  proz_abweichung.round(2)
                proz_abweichung_links[1] = proz_abweichung
                
                #Hinzufügen der Prognose zu 'df'
                df.iloc[:, 1][df.index[-1]] = nextPrediction   
                    
                #Nun soll der Graph visuell definiert werden:
                fig = px.line(title='Exponential Smoothing')
                
                fig.add_trace(go.Scatter(
                    x=df.iloc[:-1, 0],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(
                    xaxis=dict(
                        title="Zeit",
                        tickmode="linear",
                        tick0=df.iloc[0, 0],
                        dtick=2,
                        range=[df.iloc[0, 0], df.iloc[-1, 0]],
                    ),
                    yaxis=dict(
                        title="Wert"
                    )
                )


                return dcc.Graph(
                    id='exponentialsmoothing-plot',
                    figure=fig
                )

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })
    #falls der Nutzer zwei Perioden der Zukunft mithilfe des Exponential Smoothings prognostizieren lassen möchte, wird dieser Teil der Funktion 'update_output2' durchgeführt.
    elif selected_graph == 'exponentialSmoothing' and periods == '2':
        #Eingaben müssen befüllt sein, sonst kommt es zu Fehlern
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            #Zusammenführung der Inhalte in einem Pandas Dataframe 'df' (in Tabellenform)
            df = parse_contents(contents, filename)
            #Ist dieser Dataframe nun leer, bekommt der Nutzer eine rote Fehlermeldung mit einer Pixelgröße von 20 Pixeln angezeigt.
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })            
            #Ungültige Werte im Dataframe mit der Funktion 'dropna()' entfernen.
            df = df.dropna()

            #Index des Dataframes und das Dataframe 'df' selbst um zwei Reihen erweitern.
            last_index = df.iloc[-1, 0]
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            df = df.append({df.columns[0]: (last_index + 2)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                #an dieser Stelle wird kein Exponential Smoothing durchgeführt und stattdessen ein leerer Wert zurückgegeben (kein Diagramm).
                #Das liegt daran, dass die Prognose der übernächsten Periode nur auf Grundlage der letzten Prognose durchgeführt werden könnte. 
                #Alle Prognosen auf der Seite 'Forecasting im Controlling' sollen aber auf realen Vergangenheitswerten basieren.


                return ''


        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })
            
    #falls der Nutzer eine Periode der Zukunft mithilfe des Moving Average Verfahrens prognostizieren lassen möchte, wird dieser Teil der Funktion 'update_output2' durchgeführt.          
    elif selected_graph == 'movingAverage' and periods == '1':
            
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)
            
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })            
            #Dataframe 'df' von ungültigen Einträgen befreien
            df = df.dropna()
            #Dataframe um eine Reihe erweitern
            last_index = df.iloc[-1, 0]
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                #x --> Zeitachse
                #y --> die zu den Zeitwerten zugehörigen Werte
                x = df2.iloc[:, [0]].values.flatten()
                y = df2.iloc[:, [1]].values.flatten()
                
                #Moving Average der Ordnung drei berechnen
                moving_avg = moving_average(y, 3)
                nextPrediction = moving_avg[-2] 
                
                #Abweichung berechnen im Absolutbetrag
                mae = mean_absolute_error(y[-1:], moving_avg[-3:-2])
                mae = mae.round(2)
                abweichungen_links[2] = mae
                #Berechnung der prozentualen Abweichung
                proz_abweichung = (abs(moving_avg[-3:-2][0] - y[-1:][0]) / y[-1:][0]) * 100
                proz_abweichung =  proz_abweichung.round(2)
                proz_abweichung_links[2] = proz_abweichung
                
                #Hinzufügen der Prognose zu 'df'
                df.iloc[:, 1][df.index[-1]] = nextPrediction 
                
                
                #Nun soll der Graph definiert werden:
                fig = px.line(title='Moving Average (MA3)')
                
                fig.add_trace(go.Scatter(
                    x=df.iloc[:-1, 0],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(
                    xaxis=dict(
                        title="Zeit",
                        tickmode="linear",
                        tick0=df.iloc[0, 0],
                        dtick=2,
                        range=[df.iloc[0, 0], df.iloc[-1, 0]],
                    ),
                    yaxis=dict(
                        title="Wert"
                    )
                )


                return dcc.Graph(
                    id='movingaverage-plot',
                    figure=fig
                )
    

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })  
    #falls der Nutzer zwei Perioden der Zukunft mithilfe des Moving Average Verfahrens prognostizieren lassen möchte, wird dieser Teil der Funktion 'update_output2' durchgeführt.
    elif selected_graph == 'movingAverage' and periods == '2':
            
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            #Inhalte in einem Dataframe zusammenführen
            df = parse_contents(contents, filename)
            
            if df is None:
                #falls 'df' leer ist soll eine rote Fehlermeldung mit einer Pixelgröße von 20 Pixeln mittig im Block ausgegeben werden.
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })

            df = df.dropna()
            print(df.iloc[:, 0])
            last_index = df.iloc[-1, 0]
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            df = df.append({df.columns[0]: (last_index + 2)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                #an dieser Stelle wird kein Exponential Smoothing durchgeführt und stattdessen ein leerer Wert zurückgegeben (kein Diagramm).
                #Das liegt daran, dass die Prognose der übernächsten Periode nur auf Grundlage der letzten Prognose durchgeführt werden könnte. 
                #Alle Prognosen auf der Seite 'Forecasting im Controlling' sollen aber auf realen Vergangenheitswerten basieren.
                return ''
    

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })  

############rechte Seite###########################       
@app.callback(
    Output('output-data-upload', 'children'),
    Input('dropdown2', 'value'),
    Input('periods-dropdown', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(selected_graph, periods, list_of_contents, list_of_names):
    #falls der Nutzer eine Periode der Zukunft mithilfe der gewichteten, linearen Regression prognostizieren lassen möchte, wird dieser Teil der Funktion 'update_output' durchgeführt.  
    if selected_graph == 'linearRegression' and periods =='1':

        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            #Zusammenführung der eingelesenen Inhalte in einem Pandas Dataframe mit dem Namen 'df'
            df = parse_contents(contents, filename)
            #sollte die Zusammenführung fehlschlagen, wird dem Nutzer eine Fehlermeldung ausgegeben:
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
            #Löschung ungültiger Werte aus 'df'
            df = df.dropna()
            #Erweiterung von 'df' um eine Reihe
            last_index = df.iloc[-1, 0]
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                #X sei die Spalte mit den Datumsangaben und Y die zugehörigen Werte
                x = df2.iloc[:, [0]]
                y = df2.iloc[:, [1]]
                #Durchführung des Train-Test-Splits, wobei der Testdatensatz 20 % und der Trainingsdatensatz 80 % 
                #der eingelesenen (relevanten) Spalten des Datensatzes ausmachen.
                #Quelle: Hirschle, J. (2021): Machine Learning für Zeitreihen: Einstieg in Regressions-, ARIMA-und Deep Learning-Verfahren mit Python Inkl. E-Book, 1. Auflage, Carl Hanser Verlag GmbH Co. KG, München, S. 52.
                x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 0, test_size=0.2)
                #Den Trainingsdatensätze neue Indizes zuordnen (im gleichen Abstand), damit die Regression auch funktioniert.
                x_train.sort_index(inplace=True)
                y_train.sort_index(inplace=True)
                #Indizes des Trainingsdatensatzes (mit den Zeitwerten) in der Variablen 'indizes' speichern
                indizes = x_train.index.tolist()
                #Leerer Array 'gewichte' --> soll später befüllt werden
                gewichte = []
                #Anzahl der Datenpunkte entspricht der Länge des Datensatzes
                anzahlDatenpunkte = len(df2) 
                #Schleife durchlaufen lassen, die für jeden Index (=Zeitwerte) aus dem Datensatz die Gewichtung mithilfe der Tricubic Funktion berechnet.
                for x in indizes:
                    #TriCube Gewichtungsfunktion
                    d = (abs(x_train.loc[x] - x_train.iloc[-1, 0])) / anzahlDatenpunkte
                    weight = pow((1- pow(d,3)),3)
                    weight=weight.round(2)
                    weight = weight[0]
                    #Das jeweils ermittelte Gewicht wird an den Array 'gewichte' angehängt
                    gewichte.append(weight)
                #Durchführen der Linearen Regression (Modell aufsetzen und mit den Trainingsdaten sowie deren Gewichten trainieren)
                #Quelle: Hirschle, J. (2021): Machine Learning für Zeitreihen: Einstieg in Regressions-, ARIMA-und Deep Learning-Verfahren mit Python Inkl. E-Book, 1. Auflage, Carl Hanser Verlag GmbH Co. KG, München, S. 52.
                model= LinearRegression()
                model.fit(x_train, y_train, gewichte)
                #Vorhersage der zurückbehaltenen Testdaten zur anschließenden Validierung mithilfe des MAE
                y_predict = model.predict(x_test)
                #Quelle: Hirschle, J. (2021): Machine Learning für Zeitreihen: Einstieg in Regressions-, ARIMA-und Deep Learning-Verfahren mit Python Inkl. E-Book, 1. Auflage, Carl Hanser Verlag GmbH Co. KG, München, S. 52.
                mae = mean_absolute_error(y_test, y_predict)
                mae = mae.round(2)
                mae_rechts[0] = mae
                #Berechnung der Vorhersage der nächsten Periode
                nextPrediction = model.predict([[(last_index + 1)]]).round(2)
            
                #um die Vergleichbarkeit mit den herkömmlichen Forecasting-Methoden (ohne Machine Learning)
                #sicherzustellen, wird der letze Datenpunkt der Zeitreihe extrahiert, eine lineare Regr. berechnet
                #und anschließend, die Höhe des prognostizierten Werts mit dem extrahierten Wert verglichen
                
                #x2-Werte --> Zeitachse 
                #y2-Werte --> zugehörige Werte
                x2 = df2.iloc[:-1, [0]]
                y2 = df2.iloc[:-1, [1]]
                #Analoges Vorgehen wie bei der gewichteten linearen Regression zur Prognose der nächsten Periode
                #Quelle: Hirschle, J. (2021): Machine Learning für Zeitreihen: Einstieg in Regressions-, ARIMA-und Deep Learning-Verfahren mit Python Inkl. E-Book, 1. Auflage, Carl Hanser Verlag GmbH Co. KG, München, S. 52.
                x_train, x_test, y_train, y_test = train_test_split(x2, y2, random_state=0, test_size=0.2)
                x_train.sort_index(inplace=True)
                y_train.sort_index(inplace=True)

                indizes = x_train.index.tolist()

                gewichte = []
                anzahlDatenpunkte = len(df2) - 1
                
                for x in indizes:
                    #TriCube Gewichtungsfunktion
                    d = (abs(x_train.loc[x] - x_train.iloc[-1, 0])) / anzahlDatenpunkte
                    weight = pow((1- pow(d,3)),3)
                    weight=weight.round(2)
                    weight = weight[0]
                    
                    gewichte.append(weight)
                #Quelle: Hirschle, J. (2021): Machine Learning für Zeitreihen: Einstieg in Regressions-, ARIMA-und Deep Learning-Verfahren mit Python Inkl. E-Book, 1. Auflage, Carl Hanser Verlag GmbH Co. KG, München, S.52.
                #Quelle: Scikit-Learn (2023 b): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
                model = LinearRegression()
                model.fit(x_train, y_train, gewichte)
                #Um die Vergleichbarkeit mit den traditionellen Forecasting Methoden zu gewährleisten, wird die gewichtete lineare Regression nun ohne das Wissen des letzten (aktuellsten) Datenpunkts der Zeitreihe. Dieser soll dann prognostiziert werden. Die Prognose wird anschließend mit dem tatsächlichen Wert verglichen.
                #Vgl. Rabanser (2021) --> Literaturverzeichnis der Masterthesis
                verkürztePrognose = model.predict(df2.iloc[[-1], [0]])
            
                #Berechnung der Abweichung im Betrag
                abweichung = mean_absolute_error(y[-1:], verkürztePrognose)
                abweichung = abweichung.round(2)
                abweichung_rechts[0] = abweichung
                #Berechnung der prozentualen Abweichung
                proz_abweichung = (abs(verkürztePrognose[0][0] - y.iloc[-1][0]) / y.iloc[-1][0]) * 100
                proz_abweichung =  proz_abweichung.round(2)
                proz_abweichung_rechts[0] = proz_abweichung
                
                #Hinzufügen der Vorhersage zum Datensatz, damit er gleich als Diagramm dargestellt werden kann.
                df.iloc[:, 1][df.index[-1]] = nextPrediction
                
                #Nun soll der Graph definiert werden:
                fig = px.line(title='Lineare Regression (mit Machine Learning)')
                
                fig.add_trace(go.Scatter(
                    x=df.iloc[:-1, 0],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(
                    xaxis=dict(
                        title="Zeit",
                        tickmode="linear",
                        tick0=df.iloc[0, 0],
                        dtick=2,
                        range=[df.iloc[0, 0], df.iloc[-1, 0]],
                    ),
                    yaxis=dict(
                        title="Wert"
                    )
                )

                return dcc.Graph(
                    id='regression-plot',
                    figure=fig
                )
    

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })
    #falls der Nutzer zwei Perioden der Zukunft mithilfe der gewichteten, linearen Regression prognostizieren lassen möchte, wird dieser Teil der Funktion 'update_output' durchgeführt.          
    elif selected_graph == 'linearRegression' and periods =='2':
        #Analoges Vorgehen wie bei der Vorhersage einer weiteren Periode mithilfe der (gewichteten) linearen Regression. Kommentare von oben gelten hier auch.
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            df = parse_contents(contents, filename)

            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
            df = df.dropna()
            last_index = df.iloc[-1, 0]
            #Dataframe wird um zwei Reihen erweitert
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            df = df.append({df.columns[0]: (last_index + 2)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                x = df2.iloc[:, [0]]
                y = df2.iloc[:, [1]]

                #Quelle: Hirschle, J. (2021): Machine Learning für Zeitreihen: Einstieg in Regressions-, ARIMA-und Deep Learning-Verfahren mit Python Inkl. E-Book, 1. Auflage, Carl Hanser Verlag GmbH Co. KG, München, S. 52.
                x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 0, test_size=0.2)
                x_train.sort_index(inplace=True)
                y_train.sort_index(inplace=True)

                indizes = x_train.index.tolist()

                gewichte = []
                anzahlDatenpunkte = len(df2) 
                for x in indizes:
                    #TriCube Gewichtungsfunktion
                    d = (abs(x_train.loc[x] - x_train.iloc[-1, 0])) / anzahlDatenpunkte
                    weight = pow((1- pow(d,3)),3)
                    weight=weight.round(2)
                    weight = weight[0]
                    
                    gewichte.append(weight)
                #Aufsetzen und Trainieren der Linearen Regression mit Gewichtung
                #Quelle: Scikit-Learn (2023 b): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
                model= LinearRegression()
                model.fit(x_train, y_train, gewichte)
                y_predict = model.predict(x_test)

                #die zwei nächsten Jahre werden vorhergesagt
                prediction1 = model.predict([[(last_index + 1)]]).round(2)
                prediction2 = model.predict([[(last_index + 2)]]).round(2)
                #Hinzufügen der Prognosen zu 'df'
                df.iloc[:, 1][df.index[-2]] = prediction1 
                df.iloc[:, 1][df.index[-1]] = prediction2 
                
                #Nun soll der Graph definiert werden:
                fig = px.line(title='Lineare Regression (mit Machine Learning)')
                
                fig.add_trace(go.Scatter(
                    #x=df.iloc[:-1, 0],
                    #y=df.iloc[:-1, 1],
                    x=df.iloc[:-2, 0],
                    y=df.iloc[:-2, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    x=[df.iloc[-3, 0],df.iloc[-2, 0], df.iloc[-1, 0]],
                    y=[df.iloc[-3, 1],df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))

                fig.update_layout(
                    xaxis=dict(
                        title="Zeit",
                        tickmode="linear",
                        tick0=df.iloc[0, 0],
                        dtick=2,
                        range=[df.iloc[0, 0], df.iloc[-1, 0]],
                    ),
                    yaxis=dict(
                        title="Wert"
                    )
                )

                return dcc.Graph(
                    id='regression-plot',
                    figure=fig
                )
    

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })
    #falls der Nutzer eine Periode der Zukunft mithilfe des Auto-ARIMA-Verfahrens prognostizieren lassen möchte, wird dieser Teil der Funktion 'update_output' durchgeführt.            
    elif selected_graph == 'autoArima' and periods == '1':
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            #Zusammenführung der beiden eingelesenen Spalten in einem Dataframe
            df = parse_contents(contents, filename)
            
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
            #Ungültige Werte aus dem Datensatz entfernen       
            df = df.dropna()
            print(df.iloc[:, 0])
            #Dataframe um eine Reihe erweitern (dort soll dann die Prognose eingetragen werden)
            last_index = df.iloc[-1, 0]
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)

            if df is not None:
                #Entfernung ungültiger Werte aus dem Datensatz
                df2 = df.dropna()
                df2= df2.drop(df2.columns[2:], axis=1)
                print('df2:', df2)

                
                column_names = df2.columns.tolist()
                column_index = 0
                column_name = column_names[column_index]

                #df2.iloc[:, [column_index]] = pd.to_datetime(df2.iloc[:, [column_index]].astype(str).agg('-'.join, axis=1),
                #  format='%Y-%m-%d')
                df2[column_name] = pd.to_datetime(df2[column_name].astype(str).apply(lambda x: '-'.join(x.split('.0')[0].split('.') + ['01', '01'])))
            
                df2 = df2.set_index(column_name)
                #Quelle: Hirschle, J. (2021): Machine Learning für Zeitreihen: Einstieg in Regressions-, ARIMA-und Deep Learning-Verfahren mit Python Inkl. E-Book, 1. Auflage, Carl Hanser Verlag GmbH Co. KG, München, S. 118.
                grenze = 0.8 * round(len(df2))
                grenze = int(grenze)
                train = df2[:grenze]
                test = df2[grenze:]
                print('test', test)
                print('train', train)
                
                
                #Quelle: Pmdarima (2023): https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
                #Auto-Arima trainieren (jährliche Betrachtung ohne Saisonalität unter 
                # Verwendung des Augmented Dickey Fuller (ADF)Tests zur Prüfung der Stationarität)
                arima_model = auto_arima(train, start_p=0, d=1, start_q=0, test='adf', max_p=5, max_d=5, max_q=5, start_P=0,
                                         D=1, start_Q=0,max_P=5, max_D=2, max_Q=5, m =1, seasonal=False, stepwise=True, 
                                         random_state=20, n_fits=50, suppress_warnings=True, trace=True)
                
                prediction = arima_model.predict()
                print('prediction', prediction)
                
                #Vorhersage der Testreihen --> wird zunächst berechnet, 
                # um den Mean Absolute Error (MAE) berechnen zu können
                prediction = pd.DataFrame(arima_model.predict())
                prediction1 = prediction.iloc[:len(test), :] 

                #Berechnung des Mean Absolute Error zur Beurteilung der Güte 
                mae = mean_absolute_error(test, prediction1)
                mae = mae.round(2)
                mae_rechts[1] = mae
                
                #Vorhersage der nächsten Periode
                nextPrediction = arima_model.predict(int(df.index[-1]+1 - grenze))[df.index[-1] - grenze]
                nextPrediction = round(nextPrediction, 2)
    
                ##########
                #um die Vergleichbarkeit mit den herkömmlichen Forecasting-Methoden (ohne Machine Learning)
                #sicherzustellen, wird der letze Datenpunkt der Zeitreihe extrahiert, eine lineare Regr. berechnet
                #und anschließend, die Höhe des prognostizierten Werts mit dem extrahierten Wert verglichen
                
                #Um eine Reihe verkürzte Kopie von 'df2' erzeugen
                df3 = df2.iloc[:-1]
                #Quelle: Hirschle, J. (2021): Machine Learning für Zeitreihen: Einstieg in Regressions-, ARIMA-und Deep Learning-Verfahren mit Python Inkl. E-Book, 1. Auflage, Carl Hanser Verlag GmbH Co. KG, München, S. 118.
                #80 % der Daten sollen Trainingsdaten darstellen und die restlichen 20 % dienen als Testdaten
                grenze = 0.8 * round(len(df3))
                grenze = int(grenze)
                train2 = df3[:grenze]
                test2 = df3[grenze:]

                #Auto-Arima trainieren (mit den gleichen Parametern wie zuvor)
                arima_model = auto_arima(train2, start_p=0, d=1, start_q=0, test='adf', max_p=5, max_d=5, max_q=5, start_P=0, D=1, start_Q=0,
                                    max_P=5, max_D=2, max_Q=5, m =1, seasonal=False, stepwise=True, random_state=20, n_fits=50, 
                                    suppress_warnings=True, trace=True)
                #Prediction allgemein
                prediction = arima_model.predict()

                #gesuchte Vorhersage --> nächste Periode wird hier aufgrundlage von 'prediction' ermittelt
                nextPrediction2 = prediction[2] 
                nextPrediction2 = nextPrediction2.round(2)
         
                #Berechnung der Abweichung des letzten, tatsächlichen Werts zu dessen Prognose, um die Vergleichbarkeit zu herkömmlichen Forecasting-Modellen sicherzustellen.
                abweichung = mean_absolute_error(test[-1:], [nextPrediction2])
                abweichung = abweichung.round(2)
                abweichung_rechts[1] = abweichung
                #Berechnung der prozentualen Abweichung
                proz_abweichung = (abs([nextPrediction2][0] - test.iloc[-1][0]) / test.iloc[-1][0]) * 100
                proz_abweichung =  proz_abweichung.round(2)
                proz_abweichung_rechts[1] = proz_abweichung

                #Vorhersage wird nun an den Datensatz angehängt.  
                df.iloc[:, 1][df.index[-1]] = nextPrediction
            
                
                
                #Diagramm spezifizieren (Titel, Spezifikation: Welche Daten werden in blau dargestellt? Welche in Rot (=Vorhersagen)?)
                fig = px.line(title='Auto-ARIMA')
                
                fig.add_trace(go.Scatter(
                    x=df.iloc[:-1, 0],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))
                
                fig.update_layout(
                    xaxis=dict(
                        title="Zeit",
                        tickmode="linear",
                        tick0=df.iloc[0, 0],
                        dtick=2,
                        range=[df.iloc[0, 0], df.iloc[-1, 0]],
                    ),
                    yaxis=dict(
                        title="Wert"
                    )
                )


                return dcc.Graph(
                    id='arima-plot',
                    figure=fig
                )
    

         
        else:
            #falls keine Gültige Datei eingelesen wurde, soll diese rot hervorgehobene Fehlermeldung erscheinen
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })     
    #falls der Nutzer zwei Perioden der Zukunft mithilfe des Auto-ARIMA-Verfahrens prognostizieren lassen möchte, wird dieser Teil der Funktion 'update_output' durchgeführt.  
    elif selected_graph == 'autoArima' and periods == '2':
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            #Zusammenführung der eingelesenen Inhalte (der ersten beiden Spalten) in einem Pandas Dataframe in tabellarischer Form.
            df = parse_contents(contents, filename)
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
            #Entfernung ungültiger Werte
            df = df.dropna()

            last_index = df.iloc[-1, 0]
            #Datensatz 'df' um zwei Reihen erweitern
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            df = df.append({df.columns[0]: (last_index + 2)}, ignore_index=True)
 
            if df is not None:
                #ungültige Werte aus dem Datensatz entfernen
                df2 = df.dropna()
                df2= df2.drop(df2.columns[2:], axis=1)
               
                column_names = df2.columns.tolist()
                column_index = 0
                column_name = column_names[column_index]

                #erste Spalte des Dataframe in das gewünschte Datumsformat umwandeln
                #df2.iloc[:, [column_index]] = pd.to_datetime(df2.iloc[:, [column_index]].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')
                df2[column_name] = pd.to_datetime(df2[column_name].astype(str).apply(lambda x: '-'.join(x.split('.0')[0].split('.') + ['01', '01'])))
                
                df2 = df2.set_index(column_name)
                #Quelle: Hirschle, J. (2021): Machine Learning für Zeitreihen: Einstieg in Regressions-, ARIMA-und Deep Learning-Verfahren mit Python Inkl. E-Book, 1. Auflage, Carl Hanser Verlag GmbH Co. KG, München, S. 118.
                grenze = 0.8 * round(len(df2))
                grenze = int(grenze)
                train = df2[:grenze]
                test = df2[grenze:]

                #Quelle: Pmdarima (2023): https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
                #Auto-Arima trainieren
                arima_model = auto_arima(train, start_p=0, d=1, start_q=0, test='adf', max_p=5, max_d=5, max_q=5, start_P=0, D=1, start_Q=0,
                                    max_P=5, max_D=2, max_Q=5, m =1, seasonal=False, stepwise=True, random_state=20, n_fits=50, 
                                    suppress_warnings=True, trace=True)
                
                prediction = arima_model.predict()
                print('prediction', prediction)
                
                #Vorhersage der Testreihen
                prediction = pd.DataFrame(arima_model.predict())
                prediction = prediction.iloc[:len(test), :] 
                
                #Vorhersage der nächsten beiden Perioden
                prediction1 = arima_model.predict(int(df.index[-2]+1 - grenze))[df.index[-2] - grenze]
                prediction1 = round(prediction1, 2)

                prediction2 = arima_model.predict(int(df.index[-1]+1 - grenze))[df.index[-1] - grenze]
                prediction2 = round(prediction2, 2)
                
                #Anfügen der Prognosen an den Datensatz
                df.iloc[:, 1][df.index[-2]] = prediction1
                df.iloc[:, 1][df.index[-1]] = prediction2
                
                #Diagramme definieren:
                fig = px.line(title='Auto-ARIMA')
                
                fig.add_trace(go.Scatter(
                    x=df.iloc[:-2, 0],
                    y=df.iloc[:-2, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    x=[df.iloc[-3, 0],df.iloc[-2, 0], df.iloc[-1, 0]],
                    y=[df.iloc[-3, 1],df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))
                
                fig.update_layout(
                    xaxis=dict(
                        title="Zeit",
                        tickmode="linear",
                        tick0=df.iloc[0, 0],
                        dtick=2,
                        range=[df.iloc[0, 0], df.iloc[-1, 0]],
                    ),
                    yaxis=dict(
                        title="Wert"
                    )
                )


                return dcc.Graph(
                    id='arima-plot',
                    figure=fig
                )
    


        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })     
        
    #Dieser Zweig wird aufgerufen, sofern der Nutzer die Vorhersage der nächsten Periode mithilfe der Ridge Regression wünscht.   
    elif selected_graph == 'ridgeCV' and periods == '1':
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            #Inhalte in einem Pandas Dataframe 'df' speichern
            df = parse_contents(contents, filename)
            #falls 'df' leer ist, wird eine rote Fehlermeldung ausgegeben.
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
            #ungültige Werte aus 'df' entfernen
            df = df.dropna()
            #Dataframe df um eine Reihe erweitern
            last_index = df.iloc[-1, 0]
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)

            if df is not None:
                #unberechtigte Werte aus Dataframe 'df' entfernen
                df2 = df.dropna()
                #x --> Zeitwerte
                #y --> zugehörige Ausprägungen
                x = df2.iloc[:, [0]]
                y = df2.iloc[:, [1]]
                #Parameter für die Ridge Regression, die mithilfe der Cross Validation ausprobiert werden:
                myalpha = np.linspace(start = 0.1,stop = 5,num = 50)
                #Durchührung des Train-Test-Splits
                #Quelle: #Quelle: Hirschle, J. (2021): Machine Learning für Zeitreihen: Einstieg in Regressions-, ARIMA-und Deep Learning-Verfahren mit Python Inkl. E-Book, 1. Auflage, Carl Hanser Verlag GmbH Co. KG, München, S. 52.
                x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 0, test_size=0.2)
                #Ridge Regression ausführen
                ridge =RidgeCV(alphas = myalpha)
                ridge.fit(x_train, np.array(y_train).ravel())
                y_predict = ridge.predict(x_test)
                #Berechnung des Mean Absolute Error zur Evaluation der Güte des Modells
                mae = mean_absolute_error(y_test, y_predict)
                mae = mae.round(2)
                mae_rechts[2] = mae

                #Vorhersage der nächsten Periode
                nextPrediction = ridge.predict([[(last_index + 1)]]).round(2)
            
                #um die Vergleichbarkeit mit den herkömmlichen Forecasting-Methoden (ohne Machine Learning)
                #sicherzustellen, wird der letze Datenpunkt der Zeitreihe extrahiert, eine lineare Regr. berechnet
                #und anschließend, die Höhe des prognostizierten Werts mit dem extrahierten Wert verglichen
                
                #x und y Werte definieren (x bezeichnet die Zeitwerte, y die dazugehörigen Ausprägungen)
                x2 = df2.iloc[:-1, [0]]
                y2 = df2.iloc[:-1, [1]]

                x_train, x_test, y_train, y_test = train_test_split(x2, y2, random_state=0, test_size=0.2)
                #Durchführung der Ridge Regression mithilfe der Cross Validation, um das optimale Alpha zu finden.
                #Quelle: Scikit-Learn (2023 c): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
                ridge =RidgeCV(alphas = myalpha)
                #Training der Ridge Regression
                ridge.fit(x_train, np.array(y_train).ravel())
                #Verkürzte Vorhersage des letzten (verfügbaren) Datenpunkts zur späteren Berechnung der Abweichung.
                verkürztePrognose = ridge.predict(df2.iloc[[-1], [0]])
    
                #Berechnung der Abweichung des tatsächlich letzten Werts der Zeitreihe mit zum prognostizierten Wert für die letzte Reihe des eingelesenen Datensatzes
                abweichung = mean_absolute_error(y[-1:], verkürztePrognose)
                abweichung = abweichung.round(2)
                abweichung_rechts[2] = abweichung
                #Berechnung der prozentualen Abweichung
                proz_abweichung = (abs(verkürztePrognose[0] - y.iloc[-1][0]) / y.iloc[-1][0]) * 100
                proz_abweichung =  proz_abweichung.round(2)
                proz_abweichung_rechts[2] = proz_abweichung
      
                #Hinzufügen der Prognose zum Dataframe, um den Verlauf graphisch darstellen zu können
                df.iloc[:, 1][df.index[-1]] = nextPrediction

                #Nun soll der Graph definiert werden:
                fig = px.line(title='Ridge Cross Validation')
                
                fig.add_trace(go.Scatter(
                    x=df.iloc[:-1, 0],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))
                
                fig.update_layout(
                    xaxis=dict(
                        title="Zeit",
                        tickmode="linear",
                        tick0=df.iloc[0, 0],
                        dtick=2,
                        range=[df.iloc[0, 0], df.iloc[-1, 0]],
                    ),
                    yaxis=dict(
                        title="Wert"
                    )
                )


                return dcc.Graph(
                    id='ridgeCV-plot',
                    figure=fig
                )
    

        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })
    #Dieser Zweig wird aufgerufen, sofern der Nutzer die Vorhersage der nächsten zwei Perioden mithilfe der Ridge Regression wünscht.
    elif selected_graph == 'ridgeCV' and periods == '2':
        if list_of_contents is not None:
            contents = list_of_contents
            filename = list_of_names
            #Inhalte im Pandas Dataframe 'df' zusammenführen
            df = parse_contents(contents, filename)
            #falls der Dataframe leer ist, soll eine rote Fehlermeldung angezeigt werden.
            if df is None:
                return html.Div(['Die Datei scheint leer oder keine gültige CSV-Datei zu sein.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px'
                                
                                })
            #ungültige Werte aus 'df' entfernen
            df = df.dropna()
            #zwei Reihen zum Dataframe 'df' hinzufügen
            last_index = df.iloc[-1, 0]
            df = df.append({df.columns[0]: (last_index + 1)}, ignore_index=True)
            df = df.append({df.columns[0]: (last_index + 2)}, ignore_index=True)

            if df is not None:
                df2 = df.dropna()
                #x-Werte --> Zeitangaben
                x = df2.iloc[:, [0]]
                #y-Werte --> zugehörige Werte 
                y = df2.iloc[:, [1]]
                #Parameter für die Ridge Regression, die mithilfe der Cross Validation ausprobiert werden:
                myalpha = np.linspace(start = 0.1,stop = 5,num = 50)
                #Train-Test-Split durchführen. Dabei macht der Testdatensatz 20 % des Gesamtdatensatzes aus.
                #Quelle: Hirschle, J. (2021): Machine Learning für Zeitreihen: Einstieg in Regressions-, ARIMA-und Deep Learning-Verfahren mit Python Inkl. E-Book, 1. Auflage, Carl Hanser Verlag GmbH Co. KG, München, S. 52.
                x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 0, test_size=0.2)
                #Ridge Regression basierend auf der Cross Validation durchführen.
                #Quelle: Scikit-Learn (2023 c): https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
                ridge =RidgeCV(alphas = myalpha)
                ridge.fit(x_train, np.array(y_train).ravel())
                y_predict = ridge.predict(x_test)
            
                #Vorhersage der nächsten zwei Perioden
                prediction1 = ridge.predict([[(last_index + 1)]]).round(2)
                prediction2 = ridge.predict([[(last_index + 2)]]).round(2)

                #Hinzufügen der Vorhersagen zum Dataframe 
                df.iloc[:, 1][df.index[-1]] = prediction1
                df.iloc[:, 1][df.index[-1]] = prediction2

                #Nun soll der Graph definiert werden:
                fig = px.line(title='Ridge Cross Validation')
                
                fig.add_trace(go.Scatter(
                    x=df.iloc[:-1, 0],
                    y=df.iloc[:-1, 1],
                    mode="lines",
                    line=dict(color= '#4A4AE8'),
                    name="Daten"
                ))

                fig.add_trace(go.Scatter(
                    x=[df.iloc[-2, 0], df.iloc[-1, 0]],
                    y=[df.iloc[-2, 1], df.iloc[-1, 1]],
                    mode="lines",
                    line=dict(dash="dash", color="red"),
                    name="Vorhersage"
                ))
                
                fig.update_layout(
                    xaxis=dict(
                        title="Zeit",
                        tickmode="linear",
                        tick0=df.iloc[0, 0],
                        dtick=2,
                        range=[df.iloc[0, 0], df.iloc[-1, 0]],
                    ),
                    yaxis=dict(
                        title="Wert"
                    )
                )


                return dcc.Graph(
                    id='ridgeCV-plot2',
                    figure=fig
                )
    


        else:
            return html.Div(['Bitte gültige CSV-Datei hochladen.'],
                            style={
                                'textAlign': 'center',
                                'color': 'red',
                                'font-size': '20px',
                                'margin': '40px'
                                
                                })


############# Text Container Rechts ########################
#mit der Callback-Funktion wird spezifiziert, welche Inputs, Output die Funktion 'update_dropdown2_text' benötigt und welche States einen erneuten Aufruf dieser Funktion auslösen.
@app.callback(
    Output('textContainerRechts', 'children'),
    Input('dropdown2', 'value'),
    Input('periods-dropdown', 'value'),
    #Input('output-data-upload-links', 'children'),
    Input('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
    
)

def update_dropdown2_text(selected, periods, graph, x, y):
    #sofern der Nutzer die gewichtete lineare Regression und eine zu prognostizierende Periode ausgewählt hat, wird nach erfolgreicher Prüfung der unten stehende Text ausgegeben.
    if selected == 'linearRegression' and periods == '1':
        #Prüfung, ob die Inhalte der eingelesenen Datei leer sind
        if x is not None:
            contents = x
            filename = y
            #Speichern der Daten des eingelesenen Datensatzes als Pandas Dataframe
            df = parse_contents(contents, filename)
            #Sofern der Datensatz leer ist und keine Prognose durchgeführt werden kann, wird kein Erklärungstext ausgegeben
            if df is None:
                return ''
       
            else: 
                #nach erfolgreicher Berechnung der Prognose wird der untenstehende Text auf der Seite unterhalb des Diagramms ausgegeben.
                return  html.P(children=[
                    'Die ',
                    html.Strong('lineare Regression'),
                    ' der Python-Bibliothek ', html.Em('Scikit-Learn'), ' teilt den Datensatz mit dessen Spalten (bestehend aus 2 Spalten: eine für die Jahreszahlangabe und eine weitere für die zugehörigen Werte (z.B. Umsatz oder ähnliche Kennzahlen)) in Trainings-',
                    'und Testdaten auf. 20 Prozent des Datensatzes werden als Testdaten zurückbehalten. Anschließend wird die lineare Regression basierend auf den Trainingsdaten trainiert und ist nun in der Lage, die nächsten Datenpunkte für die folgenden Perioden vorherzusagen.',html.Sup('5'),
                    html.Br(), html.Br(), 'In diesem Fall wird ein Mean-Absolute-Error in Höhe von ', html.Strong(str(mae_rechts[0])), ' erzielt',
                    html.Br(), 'Dieser Mean-Absolute Error (kurz: MAE) gibt an, inwieweit die vorhergesagten Datenpunkte von den tatsächlichen Werten des Testdatensatzes abweichen. Je näher der MAE an der 0 liegt, desto besser schneidet das Machine-Learning-Modell (hier: die lineare Regression) bei der Vorhersage ab.',html.Sup('6'),
                    html.Br(), html.Br(), 'Zur besseren Vergleichbarkeit mit den traditionellen Prognosemodellen, wird die letzte Datenreihe der eingelesenen Zeitreihe im weiteren Durchgang extrahiert, noch einmal eine lineare Regression durchgeführt und der extrahierte Wert prognostiziert. Somit lässt sich erkennen, wie nah der prognostizierte Wert am tatsächlichen Wert liegt.',html.Sup('7'), ' In diesem Fall ergibt sich eine Abweichung in Höhe von:  ', html.Strong(str(abweichung_rechts[0])), '.',
                    html.Br(), html.Br(),'Des Weiteren ergibt sich eine prozentuale Abweichung des prognostizierten Werts zum realen Wert in Höhe von: ', html.Strong(str(proz_abweichung_rechts[0])), html.Strong(' %.'),
                    html.Br(),html.Br(),
                    html.Div(children=[
                        html.Sup('5'),
                        ' Scikit-Learn (2023 b).'
                    ],style={'fontSize': '12px'}),
                    html.Div(children=[
                        html.Sup('6'),
                        ' Vgl. Treyer (2010), S. 106 f.'
                    ],style={'fontSize': '12px'}),
                    html.Div(children=[
                        html.Sup('7'),
                        ' Vgl. Rabanser (2021).'
                    ],style={'fontSize': '12px'}),
                ])
     #sofern der Nutzer das Auto-ARIMA-Verfahren und eine zu prognostizierende Periode ausgewählt hat, wird nach erfolgreicher Prüfung der unten stehende Text ausgegeben.
    if selected == 'autoArima' and periods == '1':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            #Ausgabe des Textes unterhalb des Diagramms
            else: return html.P(children=[
            'Das ',
            html.Strong('Auto-ARIMA'),
            ' Modell der Python-Bibliothek ', html.Em('pmdarima'), ' teilt den Datensatz mit dessen Spalten (bestehend aus 2 Spalten: eine für die Jahreszahlangabe und eine weitere für die zugehörigen Werte (z.B. Umsatz oder ähnliche Kennzahlen)) in Trainings-',
            'und Testdaten auf. 20 Prozent des Datensatzes werden als Testdaten zurückbehalten. Anschließend erfolgt die Aufstellung eines ARIMA Zeitreihenmodells und es werden verschiedene Durchgänge mit diversen Parametern trainiert. Das Modell mit den besten Werten gewinnt und ist nun in der Lage, die nächsten Datenpunkte für die folgenden Perioden vorherzusagen.',html.Sup('8'),
            html.Br(), html.Br(), 'In diesem Fall wird ein Mean-Absolute-Error in Höhe von ', html.Strong(str(mae_rechts[1])), ' erzielt',
            html.Br(), 'Dieser Mean-Absolute Error (kurz: MAE) gibt an, inwieweit die vorhergesagten Datenpunkte von den tatsächlichen Werten des Testdatensatzes abweicht. Je näher der MAE an der 0 liegt, desto besser schneidet das Machine-Learning-Modell (hier: das Auto-ARIMA-Verfahren) bei der Vorhersage ab.',html.Sup('9'),
            html.Br(), html.Br(), 'Zur besseren Vergleichbarkeit mit den traditionellen Prognosemodellen, wird die letzte Datenreihe im weiteren Durchgang extrahiert, noch einmal eine Auto-ARIMA-Prognose durchgeführt und der extrahierte Wert prognostiziert. Somit lässt sich erkennen, wie nah der prognostizierte Wert am tatsächlichen Wert liegt.',html.Sup('10'),' In diesem Fall ergibt sich eine Abweichung in Höhe von:  ', html.Strong(str(abweichung_rechts[1])), '.',
            html.Br(), html.Br(), 'Des Weiteren ergibt sich eine prozentuale Abweichung des prognostizierten Werts zum realen Wert in Höhe von: ', html.Strong(str(proz_abweichung_rechts[1])), html.Strong(' %.'),
            html.Br(), html.Br(),
            html.Div(children=[
                html.Sup('8'),
                ' Pmdarima (2023).'
            ],style={'fontSize': '12px'}),
            html.Div(children=[
                html.Sup('9'),
                ' Vgl. Treyer (2010), S. 106 f.'
            ],style={'fontSize': '12px'}),
            html.Div(children=[
                html.Sup('10'),
                ' Vgl. Rabanser (2021).'
            ],style={'fontSize': '12px'}),
        ])
     #sofern der Nutzer die Ridge Regression und eine zu prognostizierende Periode ausgewählt hat, wird nach erfolgreicher Prüfung der unten stehende Text ausgegeben.
    if selected == 'ridgeCV' and periods == '1':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                #Ausgabe des Textes unterhalb des Diagramms
                return html.P(children=[
                    'Die ',
                    html.Strong('Ridge Regression mit Cross Validation'),
                    ' der Python-Bibliothek ', html.Em('Scikit-Learn'), ' teilt den Datensatz mit dessen Spalten (bestehend aus 2 Spalten: eine für die Jahreszahlangabe und eine weitere für die zugehörigen Werte (z.B. Umsatz oder ähnliche Kennzahlen)) in Trainings-',
                    'und Testdaten auf. 20 Prozent des Datensatzes werden als Testdaten zurückbehalten. Anschließend werden die besten Parameter automatisiert mithilfe der Cross Validation für die Ridge Regression ermittelt. Das Modell mit den besten Werten gewinnt und ist nun in der Lage, die nächsten Datenpunkte für die folgenden Perioden vorherzusagen.',html.Sup('11'),
                    html.Br(), html.Br(), 'In diesem Fall wird ein Mean-Absolute-Error in Höhe von ', html.Strong(str(mae_rechts[2])), ' erzielt',
                    html.Br(), 'Dieser Mean-Absolute Error (kurz: MAE) gibt an, inwieweit die vorhergesagten Datenpunkte von den tatsächlichen Werten des Testdatensatzes abweicht. Je näher der MAE an der 0 liegt, desto besser schneidet das Machine-Learning-Modell (hier: die Ridge Regression basierend auf der Cross Validation) bei der Vorhersage ab.',html.Sup('12'),
                    html.Br(), html.Br(), 'Zur besseren Vergleichbarkeit mit den traditionellen Prognosemodellen, wird die letzte Datenreihe im weiteren Durchgang extrahiert, noch einmal eine Ridge Regression mit Cross Validation durchgeführt und der extrahierte Wert prognostiziert. Somit lässt sich erkennen, wie nah der prognostizierte Wert am tatsächlichen Wert liegt.',html.Sup('13'), ' In diesem Fall ergibt sich eine Abweichung in Höhe von: ', html.Strong(str(abweichung_rechts[2])), '.',
                    html.Br(), html.Br(), 'Des Weiteren ergibt sich eine prozentuale Abweichung des prognostizierten Werts zum realen Wert in Höhe von: ', html.Strong(str(proz_abweichung_rechts[2])), html.Strong(' %.'),
                    html.Br(),html.Br(),
                    html.Div(children=[
                        html.Sup('11'),
                        ' Scikit-Learn (2023 c).'
                    ],style={'fontSize': '12px'}),
                    html.Div(children=[
                        html.Sup('12'),
                        ' Vgl. Treyer (2010), S. 106 f.'
                    ],style={'fontSize': '12px'}),
                    html.Div(children=[
                        html.Sup('13'),
                        ' Vgl. Rabanser (2021).'
                    ],style={'fontSize': '12px'}),
                ])
     #sofern der Nutzer die gewichtete lineare Regression und zwei zu prognostizierende Perioden ausgewählt hat, wird nach erfolgreicher Prüfung der unten stehende Text ausgegeben.
    if selected == 'linearRegression' and periods == '2':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
       
            else: 
                return  html.P(children=[
                    'Die ',
                    html.Strong('lineare Regression'),
                    ' der Python-Bibliothek ', html.Em('Scikit-Learn'), ' teilt den Datensatz mit dessen Spalten (bestehend aus 2 Spalten: eine für die Jahreszahlangabe und eine weitere für die zugehörigen Werte (z.B. Umsatz oder ähnliche Kennzahlen)) in Trainings-',
                    'und Testdaten auf. 20 Prozent des Datensatzes werden als Testdaten zurückbehalten. Anschließend wird die lineare Regression basierend auf den Trainingsdaten trainiert und ist nun in der Lage, die nächsten Datenpunkte für die folgenden Perioden vorherzusagen.',html.Sup('5'),
                    html.Br(), html.Br(), 'In diesem Fall wird ein Mean-Absolute-Error in Höhe von ', html.Strong(str(mae_rechts[0])), ' erzielt',
                    html.Br(), 'Dieser Mean-Absolute Error (kurz: MAE) gibt an, inwieweit die vorhergesagten Datenpunkte von den tatsächlichen Werten des Testdatensatzes abweichen. Je näher der MAE an der 0 liegt, desto besser schneidet das Machine-Learning-Modell (hier: die lineare Regression) bei der Vorhersage ab.',html.Sup('6'),
                    html.Br(), html.Br(), 'Zur besseren Vergleichbarkeit mit den traditionellen Prognosemodellen, wird die letzte Datenreihe der eingelesenen Zeitreihe im weiteren Durchgang extrahiert, noch einmal eine lineare Regression durchgeführt und der extrahierte Wert prognostiziert. Somit lässt sich erkennen, wie nah der prognostizierte Wert am tatsächlichen Wert liegt.',html.Sup('7'), ' In diesem Fall ergibt sich eine Abweichung in Höhe von:  ', html.Strong(str(abweichung_rechts[0])), '.',
                    html.Br(), html.Br(),'Des Weiteren ergibt sich eine prozentuale Abweichung des prognostizierten Werts zum realen Wert in Höhe von: ', html.Strong(str(proz_abweichung_rechts[0])), html.Strong(' %.'),
                    html.Br(),html.Br(),
                    html.Div(children=[
                        html.Sup('5'),
                        ' Scikit-Learn (2023 b).'
                    ],style={'fontSize': '12px'}),
                    html.Div(children=[
                        html.Sup('6'),
                        ' Vgl. Treyer (2010), S. 106 f.'
                    ],style={'fontSize': '12px'}),
                    html.Div(children=[
                        html.Sup('7'),
                        ' Vgl. Rabanser (2021).'
                    ],style={'fontSize': '12px'}),
                ])
     #sofern der Nutzer das Auto-ARIMA-Verfahren und zwei zu prognostizierende Perioden ausgewählt hat, wird nach erfolgreicher Prüfung der unten stehende Text ausgegeben.
    if selected == 'autoArima' and periods == '2':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            #Ausgabe des Textes unterhalb des Diagramms
            else: return html.P(children=[
            'Das ',
            html.Strong('Auto-ARIMA'),
            ' Modell der Python-Bibliothek ', html.Em('pmdarima'), ' teilt den Datensatz mit dessen Spalten (bestehend aus 2 Spalten: eine für die Jahreszahlangabe und eine weitere für die zugehörigen Werte (z.B. Umsatz oder ähnliche Kennzahlen)) in Trainings-',
            'und Testdaten auf. 20 Prozent des Datensatzes werden als Testdaten zurückbehalten. Anschließend erfolgt die Aufstellung eines ARIMA Zeitreihenmodells und es werden verschiedene Durchgänge mit diversen Parametern trainiert. Das Modell mit den besten Werten gewinnt und ist nun in der Lage, die nächsten Datenpunkte für die folgenden Perioden vorherzusagen.',html.Sup('8'),
            html.Br(), html.Br(), 'In diesem Fall wird ein Mean-Absolute-Error in Höhe von ', html.Strong(str(mae_rechts[1])), ' erzielt',
            html.Br(), 'Dieser Mean-Absolute Error (kurz: MAE) gibt an, inwieweit die vorhergesagten Datenpunkte von den tatsächlichen Werten des Testdatensatzes abweicht. Je näher der MAE an der 0 liegt, desto besser schneidet das Machine-Learning-Modell (hier: das Auto-ARIMA-Verfahren) bei der Vorhersage ab.',html.Sup('9'),
            html.Br(), html.Br(), 'Zur besseren Vergleichbarkeit mit den traditionellen Prognosemodellen, wird die letzte Datenreihe im weiteren Durchgang extrahiert, noch einmal eine Auto-ARIMA-Prognose durchgeführt und der extrahierte Wert prognostiziert. Somit lässt sich erkennen, wie nah der prognostizierte Wert am tatsächlichen Wert liegt.',html.Sup('10'),' In diesem Fall ergibt sich eine Abweichung in Höhe von:  ', html.Strong(str(abweichung_rechts[1])), '.',
            html.Br(), html.Br(), 'Des Weiteren ergibt sich eine prozentuale Abweichung des prognostizierten Werts zum realen Wert in Höhe von: ', html.Strong(str(proz_abweichung_rechts[1])), html.Strong(' %.'),
            html.Br(), html.Br(),
            html.Div(children=[
                html.Sup('8'),
                ' Pmdarima (2023).'
            ],style={'fontSize': '12px'}),
            html.Div(children=[
                html.Sup('9'),
                ' Vgl. Treyer (2010), S. 106 f.'
            ],style={'fontSize': '12px'}),
            html.Div(children=[
                html.Sup('10'),
                ' Vgl. Rabanser (2021).'
            ],style={'fontSize': '12px'}),
        ])
     #sofern der Nutzer die Ridge Regression und zwei zu prognostizierende Perioden ausgewählt hat, wird nach erfolgreicher Prüfung der unten stehende Text ausgegeben.
    if selected == 'ridgeCV' and periods == '2':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                #Ausgabe des Textes unterhalb des Diagramms
                return html.P(children=[
                    'Die ',
                    html.Strong('Ridge Regression mit Cross Validation'),
                    ' der Python-Bibliothek ', html.Em('Scikit-Learn'), ' teilt den Datensatz mit dessen Spalten (bestehend aus 2 Spalten: eine für die Jahreszahlangabe und eine weitere für die zugehörigen Werte (z.B. Umsatz oder ähnliche Kennzahlen)) in Trainings-',
                    'und Testdaten auf. 20 Prozent des Datensatzes werden als Testdaten zurückbehalten. Anschließend werden die besten Parameter automatisiert mithilfe der Cross Validation für die Ridge Regression ermittelt. Das Modell mit den besten Werten gewinnt und ist nun in der Lage, die nächsten Datenpunkte für die folgenden Perioden vorherzusagen.',html.Sup('11'),
                    html.Br(), html.Br(), 'In diesem Fall wird ein Mean-Absolute-Error in Höhe von ', html.Strong(str(mae_rechts[2])), ' erzielt',
                    html.Br(), 'Dieser Mean-Absolute Error (kurz: MAE) gibt an, inwieweit die vorhergesagten Datenpunkte von den tatsächlichen Werten des Testdatensatzes abweicht. Je näher der MAE an der 0 liegt, desto besser schneidet das Machine-Learning-Modell (hier: die Ridge Regression basierend auf der Cross Validation) bei der Vorhersage ab.',html.Sup('12'),
                    html.Br(), html.Br(), 'Zur besseren Vergleichbarkeit mit den traditionellen Prognosemodellen, wird die letzte Datenreihe im weiteren Durchgang extrahiert, noch einmal eine Ridge Regression mit Cross Validation durchgeführt und der extrahierte Wert prognostiziert. Somit lässt sich erkennen, wie nah der prognostizierte Wert am tatsächlichen Wert liegt.',html.Sup('13'), ' In diesem Fall ergibt sich eine Abweichung in Höhe von: ', html.Strong(str(abweichung_rechts[2])), '.',
                    html.Br(), html.Br(), 'Des Weiteren ergibt sich eine prozentuale Abweichung des prognostizierten Werts zum realen Wert in Höhe von: ', html.Strong(str(proz_abweichung_rechts[2])), html.Strong(' %.'),
                    html.Br(),html.Br(),
                    html.Div(children=[
                        html.Sup('11'),
                        ' Scikit-Learn (2023 c).'
                    ],style={'fontSize': '12px'}),
                    html.Div(children=[
                        html.Sup('12'),
                        ' Vgl. Treyer (2010), S. 106 f.'
                    ],style={'fontSize': '12px'}),
                    html.Div(children=[
                        html.Sup('13'),
                        ' Vgl. Rabanser (2021).'
                    ],style={'fontSize': '12px'}),
                ])

    else:
        return ''
############# Text Container Links ########################
#mit der Callback-Funktion wird spezifiziert, welche Inputs, Output die Funktion 'update_dropdown2_text' benötigt und welche States einen erneuten Aufruf dieser Funktion auslösen.
@app.callback(
    Output('textContainerLinks', 'children'),
    Input('dropdown1', 'value'),
    Input('periods-dropdown', 'value'),
    Input('output-data-upload-links', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)

def update_dropdown2_text(selected, periods, graph, x, y):
    #sofern der Nutzer die klassische lineare Regression und eine zu prognostizierende Periode ausgewählt hat, wird nach erfolgreicher Prüfung der unten stehende Text ausgegeben.
    if selected == 'linReg' and periods =='1':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                #Ausgabe des Textes unterhalb des Diagramms
                return html.P(children=[
                    'Dieser Graph zeigt die Ergebnisse der ',
                    html.Strong('klassischen linearen Regression'),
                    ', welche mithilfe der Python-Bibliothek Scipy durchgeführt wird.', html.Sup('1'),html.Br(), html.Br(), 'Um die Güte dieses Modells vergleichen zu können, wird abermals eine lineare Regression ohne die letzte Datenreihe durchgeführt und der extrahierte Wert prognostiziert. Anschließend kann mit der Differenz des tatsächlichen Werts zum prognostizierten Wert beurteilt werden, wie gut das Modell bei der Vorhersage abschneidet.', html.Sup('2'),' In diesem Fall ergibt sich eine Abweichung in Höhe von: ', html.Strong(str(abweichungen_links[0])), '.',
                    html.Br(),html.Br(), 'Des Weiteren ergibt sich eine prozentuale Abweichung des prognostizierten Werts zum realen Wert in Höhe von: ', html.Strong(str(proz_abweichung_links[0])), html.Strong(' %.'),
                    html.Br(),html.Br(),
                        html.Div(children=[
                            html.Sup('1'),
                            ' Vgl. Scipy (o.D. b).'
                        ],style={'fontSize': '12px'}),
                        html.Div(children=[
                            html.Sup('2'),
                            ' Vgl. Rabanser (2021).'
                        ],style={'fontSize': '12px'}),
                ])
    #sofern der Nutzer das Exponential Smoothing und eine zu prognostizierende Periode ausgewählt hat, wird nach erfolgreicher Prüfung der unten stehende Text ausgegeben.
    if selected == 'exponentialSmoothing' and periods =='1':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                #Ausgabe des Textes unterhalb des Diagramms
                return html.P(children=[
                    'Dieser Graph zeigt die Ergebnisse des ',
                    html.Strong('Expontential Smoothing'),
                    ' (Exponentielle Glättung). ',html.Br(), html.Br(), 'Um die Güte dieses Modells vergleichen zu können, wird abermals eine exponentielle Glättung ohne die letzte Datenreihe durchgeführt und der extrahierte Wert prognostiziert. Anschließend kann mit der Differenz des tatsächlichen Werts zum prognostizierten Wert beurteilt werden, wie gut das Modell bei der Vorhersage abschneidet.',html.Sup('4'),' In diesem Fall ergibt sich eine Abweichung in Höhe von: ', html.Strong(str(abweichungen_links[1])), '.',
                    html.Br(),html.Br(), 'Des Weiteren ergibt sich eine prozentuale Abweichung des prognostizierten Werts zum realen Wert in Höhe von: ', html.Strong(str(proz_abweichung_links[1])), html.Strong(' %.'),
                    html.Br(),html.Br(),
                    html.Div(children=[
                        html.Sup('4'),
                        ' Vgl. Rabanser (2021).'
                    ],style={'fontSize': '12px'}),
                    
                ])
    #sofern der Nutzer das Moving-Average-Verfahren und eine zu prognostizierende Periode ausgewählt hat, wird nach erfolgreicher Prüfung der unten stehende Text ausgegeben.
    if selected == 'movingAverage'and periods =='1':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                #Ausgabe des Textes unterhalb des Diagramms
                return html.P(children=[
                    'Dieser Graph zeigt die Ergebnisse des ',
                    html.Strong('Moving Average Modell der dritten Ordnung'),
                    '. Das heißt basierend auf den jeweils drei vorherigen Datenpunkten wird die nächste Periode vorhergesagt.',
                    html.Br(), html.Br(), 'Um die Güte dieses Modells vergleichen zu können, wird abermals ein Moving-Average Verfahren ohne die letzte Datenreihe durchgeführt und der extrahierte Wert prognostiziert. Anschließend kann mit der Differenz des tatsächlichen Werts zum prognostizierten Wert beurteilt werden, wie gut das Modell bei der Vorhersage abschneidet.',html.Sup('3'), ' In diesem Fall ergibt sich eine Abweichung in Höhe von: ', html.Strong(str(abweichungen_links[2])), '.',
                    html.Br(), html.Br(),'Des Weiteren ergibt sich eine prozentuale Abweichung des prognostizierten Werts zum realen Wert in Höhe von: ', html.Strong(str(proz_abweichung_links[2])), html.Strong(' %.'),
                    html.Br(), html.Br(),
                    html.Div(children=[
                        html.Sup('3'),
                        ' Vgl. Rabanser (2021).'
                    ],style={'fontSize': '12px'}),
                ])
    #sofern der Nutzer die klassische lineare Regression und zwei zu prognostizierende Perioden ausgewählt hat, wird nach erfolgreicher Prüfung der unten stehende Text ausgegeben.
    if selected == 'linReg' and periods =='2':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                #Ausgabe des Textes unterhalb des Diagramms
                return html.P(children=[
                    'Dieser Graph zeigt die Ergebnisse der ',
                    html.Strong('klassischen linearen Regression'),
                    ' ,welche mithilfe der Python-Bibliothek Scipy durchgeführt wird.', html.Sup('1'), html.Br(), html.Br(), 'Um die Güte dieses Modells vergleichen zu können, wird abermals eine lineare Regression ohne die letzte Datenreihe durchgeführt und der extrahierte Wert prognostiziert. Anschließend kann mit der Differenz des tatsächlichen Werts zum prognostizierten Wert beurteilt werden, wie gut das Modell bei der Vorhersage abschneidet.',html.Sup('2'), ' In diesem Fall ergibt sich eine Abweichung in Höhe von: ', html.Strong(str(abweichungen_links[0])), '.',
                    html.Br(),html.Br(), 'Des Weiteren ergibt sich eine prozentuale Abweichung des prognostizierten Werts zum realen Wert in Höhe von: ', html.Strong(str(proz_abweichung_links[0])), html.Strong(' %.'),
                    html.Br(),html.Br(),
                    html.Div(children=[
                        html.Sup('1'),
                        ' Vgl. Scipy (o.D. b).'
                    ],style={'fontSize': '12px'}),
                    html.Div(children=[
                        html.Sup('2'),
                        ' Vgl. Rabanser (2021).'
                    ],style={'fontSize': '12px'}),
                ])
    #sofern der Nutzer das Exponential Smoothing und zwei zu prognostizierende Perioden ausgewählt hat, wird nach erfolgreicher Prüfung der unten stehende Text ausgegeben.
    if selected == 'exponentialSmoothing' and periods =='2':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                #Ausgabe des Hinweistextes
                return html.P(children=[
                    'Das ',
                    html.Strong('Expontential Smoothing'),
                    ' (Exponentielle Glättung) ist nicht in der Lage auf (reiner) Grundlage von Vergangenheitsdaten mehr als eine Periode zu prognostizieren. Deswegen wird an dieser Stelle kein Graph ausgegeben (siehe Kapitel 4.1.3.3 der Masterthesis).',
                ])
    #sofern der Nutzer das Moving Average Verfahren und zwei zu prognostizierende Perioden ausgewählt hat, wird nach erfolgreicher Prüfung der unten stehende Text ausgegeben.
    if selected == 'movingAverage' and periods =='2':
        if x is not None:
            contents = x
            filename = y
            df = parse_contents(contents, filename)

            if df is None:
                return ''
            else:
                #Ausgabe des Hinweistextes 
                return html.P(children=[
                    'Das ',
                    html.Strong('Moving Average Modell der dritten Ordnung'),
                    ' . (MA(3)) ist nicht in der Lage auf (reiner) Grundlage von Vergangenheitsdaten mehr als eine Periode zu prognostizieren. Deswegen wird an dieser Stelle kein Graph ausgegeben (siehe Kapitel 4.1.3.2 der Masterthesis).',
                ])
  
    else:
        return ''

app.layout = layout
#Nun starte ich die App 
if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
