import streamlit as st
import pandas as pd
import numpy as np
import modules.plots as plots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

st.set_page_config(layout="wide")

df = pd.read_csv('data/Yelp_business_data_prediction.csv')

# Two-way graphs
response = st.selectbox('Select response', ['satisfied', 'unsatisfied', 'stars'], index=0)
fitted_values = st.selectbox('Select prediction', ['gbm_predicted', 'rf_predicted', 'logit_predicted', 'gbm_predicted_bin', 'rf_predicted_bin', 'logit_predicted_bin'])

colnames = list(df.columns)
variable_options = [col for col in colnames if col != response]
variable1 = st.selectbox('First variable', variable_options, index=7)
variable2 = st.selectbox('Second variable', [''] + variable_options, index=0)

if variable2:
    fig = plots.investigate_categoric_variable(df, response, [variable1, variable2], ['Unknown'], False, fitted_values)
else:
    fig = plots.investigate_categoric_variable(df, response, [variable1], ['Unknown'], False, fitted_values)
    
fig.update_layout(height=700)

st.plotly_chart(fig, use_container_width=True, height=700)

# Geographic visualization
map_type = st.selectbox('Select type of map:', ['Scatter Points Map', 'Heatmap Map', 'Hexa Map'], index=2)

if map_type == 'Scatter Points Map':
    
    map_variables = st.multiselect("Select variables to display in hover on map point:", variable_options)
    map_color_variable = st.selectbox('Select variable to color map:', variable_options, index=2)
    bottom_record, top_record = st.slider('Limit number of records displayed in map:', 
                                        min_value=0, 
                                        max_value=df.shape[0], 
                                        value=(0, df.shape[0]))
    color_scale = [(0, 'orange'), (1,'red')]

    fig_geo = px.scatter_mapbox(
        df.iloc[bottom_record:(top_record+1)], 
        lat="latitude",
        lon="longitude", 
        hover_name="name",
        hover_data=map_variables,
        color=map_color_variable,
        color_continuous_scale='thermal',
        size='review_count',
        zoom=3, 
        mapbox_style="open-street-map",
        height=800,
        width=800)
    fig_geo.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
elif map_type == 'Heatmap Map':
    
    fig_geo = px.density_mapbox(
        df, 
        lat='latitude', 
        lon='longitude', 
        z='review_count',
        color_continuous_scale='thermal', 
        radius=10,
        zoom=3,
        mapbox_style="stamen-terrain")
    fig_geo.update_layout(mapbox_style="open-street-map")
    fig_geo.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
else:
    
    variable_options_hexa = ['stars', 'is_open', 'RestaurantsPriceRange2', 'RestaurantsPriceRange2_Unknown', 
                             'avg_time_open_week', 'avg_time_open_weekend', 'avg_time_open_week_Unknown', 'avg_time_open_weekend_Unknown',
                             'gbm_predicted', 'rf_predicted', 'logit_predicted']
    no_hexagons = st.slider('Number of hexagons:', min_value=100, max_value=2000, value=1000)
    map_variable = st.selectbox('Select variable for map:', variable_options_hexa, index=0)
    show_data = st.selectbox('Show underlying data:', ['True', 'False'], index=1)
    if show_data == "True":
        bool_data = True
    else:
        bool_data = False
    fig_geo = ff.create_hexbin_mapbox(
        data_frame=df, 
        lat="latitude", 
        lon="longitude",
        nx_hexagon=no_hexagons,
        opacity=0.5, 
        labels={"color": map_variable},
        color=map_variable,
        color_continuous_scale='thermal',
        mapbox_style="open-street-map",
        agg_func=np.mean,
        show_original_data=bool_data,
        original_data_marker=dict(size=4, opacity=0.9, color="deeppink"),
        zoom=3,
        min_count=1)
    fig_geo.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

st.plotly_chart(fig_geo, use_container_width=True, height=700)
