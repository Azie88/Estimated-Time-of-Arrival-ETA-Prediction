import gradio as gr
import numpy as np
import pandas as pd
import os, joblib
import re

# load model pipeline
file_path = os.path.abspath('toolkit/pipeline.joblib')
pipeline = joblib.load(file_path)


#function to calculate week hour from weekday and hour
def calculate_pickup_week_hour(pickup_hour, pickup_weekday):
    return pickup_weekday * 24 + pickup_hour

def predict(origin_lat, origin_lon, Destination_lat, Destination_lon,
            Trip_distance, dewpoint_2m_temperature,
            minimum_2m_air_temperature, pickup_weekday, pickup_hour,
            cluster_id, temperature_range, rain):
    
    # Calculate pickup_week_hour
    pickup_week_hour = calculate_pickup_week_hour(pickup_hour, pickup_weekday)

    # Modeling
    try:
        model_output = abs(int(pipeline.predict(pd.DataFrame([[origin_lat, origin_lon, Destination_lat, Destination_lon,
                                                               Trip_distance, dewpoint_2m_temperature,
                                                               minimum_2m_air_temperature, pickup_weekday, pickup_hour,
                                                               pickup_week_hour, cluster_id, temperature_range,
                                                               rain]], columns=['Origin_lat', 'Origin_lon', 'Destination_lat',
                                                                                'Destination_lon', 'Trip_distance',
                                                                                'dewpoint_2m_temperature',
                                                                                'minimum_2m_air_temperature',
                                                                                'pickup_weekday', 'pickup_hour',
                                                                                'pickup_week_hour', 'cluster_id',
                                                                                'temperature_range', 'rain']))))
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        model_output = 0

    output_str = 'Hey there, Your ETA is'
    dist = 'seconds'

    return f"{output_str} {model_output} {dist}"

# UI layout
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# ETA PREDICTION")
    gr.Markdown("""This app uses a machine learning model to predict the ETA of trips on the Yassir Hailing App.Refer to the expander at the bottom for more information on the inputs.""")

    with gr.Row():
        origin_lat = gr.Slider(2.806, 3.373, step=0.001, interactive=True, value=2.806, label='Origin latitude')
        origin_lon = gr.Slider(36.589, 36.820, step=0.001, interactive=True, value=36.589, label='Origin longitude')
        Destination_lat = gr.Slider(2.807, 3.381, step=0.001, interactive=True, value=2.810, label='Destination latitude')
        Destination_lon = gr.Slider(36.592, 36.819, step=0.001, interactive=True, value=36.596, label='Destination longitude')
        Trip_distance = gr.Slider(0, 62028, step=1, interactive=True, value=1000, label='Trip distance (M)')
        cluster_id = gr.Dropdown([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Cluster ID", value=4)

    with gr.Column():
        pickup_weekday = gr.Dropdown([0, 1, 2, 3, 4, 5, 6], value=3, label='Pickup weekday')
        pickup_hour = gr.Dropdown([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                                   value=13, label='Pickup hour')

    with gr.Column():
        dewpoint_2m_temperature = gr.Slider(279.129, 286.327, step=0.001, interactive=True, value=282.201,
                                             label='dewpoint_2m_temperature')
        minimum_2m_air_temperature = gr.Slider(282.037, 292.543, step=0.01, interactive=True, value=285.203,
                                               label='minimum_2m_air_temperature')
        temperature_range = gr.Slider(1.663, 10.022, step=0.01, interactive=True, value=5, label='temperature_range')
        rain = gr.Dropdown([0, 1], label='Is it raining (0=No, 1=Yes)')

    with gr.Row():
        btn = gr.Button("Predict")
        output = gr.Textbox(label="Prediction")

    # Expander for more info on columns
    with gr.Accordion("Information on inputs"):
        gr.Markdown("""These are information on the inputs the app takes for predicting a rides ETA.
                    - Origin latitude: Origin in degree latitude)
                    - Origin longitude:  Origin in degree longitude
                    - Destination latitude: Destination latitude
                    - Destination longitude: Destination logitude
                    - Trip distance (M): Distance in meters on a driving route
                    - Cluster ID: Select the cluster within which you started your trip
                    - Pickup weekday: Day of the week
                        Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6
                    - Pickup hour: The hour of the day (24hr clock)
                    - dewpoint_2m_temperature: The temperature at 2 meters above the ground where the air temperature would be 
                      low enough for dew to form. It gives an indication of humidity.
                    - minimum_2m_air_temperature: The lowest air temperature recorded at 2 meters above the ground during the specified date.
                    - temperature_range: The air temperature range recorded at 2 meters above the ground on the day
                    - rain: Is it raining? yes=1, no=2
                    """)

    btn.click(fn=predict, inputs=[origin_lat, origin_lon, Destination_lat, Destination_lon,
                                  Trip_distance, dewpoint_2m_temperature,
                                  minimum_2m_air_temperature, pickup_weekday, pickup_hour,
                                  cluster_id, temperature_range,
                                  rain], outputs=output)

    app.launch(share=True, debug=True)
