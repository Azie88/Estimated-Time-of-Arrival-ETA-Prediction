# Import the required Libraries
import gradio as gr
import numpy as np
import pandas as pd
import os, joblib
import re


pipeline = joblib.load(r'toolkit\pipeline.joblib')


inputs = ['Origin_lat', 'Origin_lon', 'Destination_lat', 'Destination_lon',
       'Trip_distance', 'Speed', 'dewpoint_2m_temperature',
       'mean_2m_air_temperature', 'pickup_weekday', 'pickup_weekofyear',
       'pickup_hour', 'pickup_minute', 'pickup_week_hour', 'month',
       'day_of_week', 'cluster_id', 'target_transformed', 'temperature_range',
       'wind_speed', 'rain']

def predict(*args, pipeline=pipeline, inputs=inputs):
    # Check if inputs is provided
    if inputs is None:
        raise ValueError("Please provide the 'inputs' parameter.")

    # Creating a DataFrame of inputs
    input_data = pd.DataFrame([args], columns=inputs)
    print(input_data)

    # Modeling
    try:
        model_output = abs(int(pipeline.predict(input_data)))
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        model_output = 0

    output_str = 'Hey there, Your ETA is'
    dist = 'seconds'

    return f"{output_str} {model_output} {dist}"

   
with gr.Blocks(theme=gr.themes.Monochrome()) as app:
  gr.Markdown("# ETA PREDICTION")
  gr.Markdown("""This app uses a machine learning model to predict the ETA of trips on the Yassir Hailing App.Refer to the expander at the bottom for more information on the inputs.""")

  with gr.Row():
        origin_lat= gr.Slider(2.806,3.381,step = 0.01,interactive=True, value=2.806, label = 'origin_lat')
        origin_lon = gr.Slider(36.589,36.820,step =0.01,interactive=True, value=36.589,label = 'origin_lon')
        Destination_lat =gr.Slider(2.807,3.381,step = 0.1,interactive=True, value=2.81,label ='Destination_lat')
        Destination_lon =gr.Slider(36.592,36.819,step = 0.1,interactive=True, value=36.596,label ='Destination_lon')
        Trip_distance = gr.Slider(61,958,step =10,interactive=True, value=20,label = 'Trip_distance')
        Speed = gr.Slider(0.300,85.000, step=0.01, interactive=True, value=20, label = 'Speed')
        
  with gr.Column():
    dewpoint_2m_temperature =gr.Slider(280.000, 288.000, step = 0.01,interactive=True, value=282.201,label ='dewpoint_2m_temperature')
    mean_2m_air_temperature =gr.Slider(285.203, 291.110,step = 0.1,interactive=True, value=287.203,label ='mean_2m_air_temperature')
    pickup_weekday = gr.Dropdown([0,1,2,3,4,5,6],label ='pickup_weekday', value=3)
    pickup_weekofyear = gr.Dropdown([47,48,49,50],label="pickup_weekofyear",value=3)
    pickup_hour = gr.Dropdown([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
                               ,label="pickup_hour",value=13)
    pickup_minute = gr.Slider(0, 59, step = 1,interactive=True, value=5,label ='pickup_minute')
    pickup_week_hour = gr.Slider(0, 167, step = 1,interactive=True, value=5,label ='pickup_week_hour')
    month = gr.Dropdown([11,12],label = 'month')
    day_of_week = gr.Dropdown([0,1,2,3,4,5,6],label ='day_of_week', value=3)
    cluster_id = gr.Dropdown([1,2,3,4,5,6,7],label="Cluster ID", value=4)

  with gr.Column():
    target_transformed = gr.Slider(0, 336.426, step = 0.01,interactive=True, value=100,label ='target_transformed')
    temperature_range = gr.Slider(2.317, 9.166, step = 0.01,interactive=True, value=5,label='temperature_range')
    wind_speed = gr.Slider(0.803, 9.887, step = 0.01,interactive=True, value=5,label='wind_speed')
    rain = gr.Dropdown([0,1],label='rain')

  with gr.Row():
    btn = gr.Button("Predict")
    output = gr.Textbox(label="Prediction")

    # Expander for more info on columns
  with gr.Accordion("Information on inputs"):
      gr.Markdown("""These are information on the inputs the app takes for predicting a rides ETA.
                    - origin_lat: Origin in degree latitude)
                    - origin_lon:  Origin in degree longitude
                    - Destination_lat: Destination latitude
                    - Destination_lon: Destination logitude
                    - Trip Distance : Distance in meters on a driving route
                    - Cluster ID : Select the cluster within which you started your trip
                    - Time of the day: What time in the day did your trip start, 1- morning(or daytime),2 - evening 3- midnight
                    """)
  btn.click(fn = predict,inputs= [origin_lat, origin_lon, Destination_lat, Destination_lon,
       Trip_distance, Speed, dewpoint_2m_temperature,
       mean_2m_air_temperature, pickup_weekday, pickup_weekofyear,
       pickup_hour, pickup_minute, pickup_week_hour, month,
       day_of_week, cluster_id, target_transformed, temperature_range,
       wind_speed, rain], outputs = output)
  
  app.launch(share = True, debug =True)