import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path

import os

import pickle
import pandas as pd
import csv
import datetime as dt
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import cv2




# import datasets

# import dataset 
final_path = Path(__file__).parent / 'data/combined_coords.csv'
df = pd.read_csv(final_path)

# import audio files
audio_path = Path(__file__).parent / 'audio_clips/'

# import video files


# load the model
model_path = Path(__file__).parent / 'models/muaythai.pkl'
model = pickle.load(open(model_path, 'rb'))



# streamlit shell (layouts etc)
# set webpage name and icon
st.set_page_config(
    page_title='Muay Th.AI Trainer',
    page_icon=':boxing_glove:',
    layout='wide',
    initial_sidebar_state='expanded'
    )

# top navigation bar
selected = option_menu(
    menu_title = None,
    options = ['About','Live Muay Th.AI', 'Upload A Video'],
    icons = ['eyeglasses','camera-reels-feel','collection-play-fill'],
    default_index = 0, # which tab it should open when page is first loaded
    orientation = 'horizontal',
    styles={
        'nav-link-selected': {'background-color': '#FF7F0E'}
        }
    )

if selected == 'About':
    # title
    st.title('About')
    st.subheader('by Wynne Chen')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)



    # comparative bar/line chart 
    # ask for user input
    st.subheader('What is Muay Thai?')
    
    st.video()
    
    
    option = st.selectbox('Pick a bodypart to see how it moves in the jab vs the kick',
                            ('Left Eye', 'Left Hand', ))
    
    # translate the english from the option box into the equivalent variable
    if option == 'Precipitation':
        variable = 'roll_sum_28_PrecipTotal'
    elif option == 'Average Temperature':
        variable = 'roll_mean_28_Tavg'

    

    # create the dataframes for the plots    
    j = df[df['class']=='jab']
    k = df[df['class']=='kick']
    
    # draw the line graph for the chosen variable
    plt.figure(figsize = (8,8))

    sns.scatterplot(x = df1[df1['left_wrist_conf']>0.2]['left_wrist_x'], y = 1 - df1[df1['left_wrist_conf']>0.3]['left_wrist_y'], alpha = 0.2, color = 'g', label = 'jabs')
    sns.scatterplot(x = df2[df2['left_wrist_conf']>0.2]['left_wrist_x'], y = 1 - df2[df2['left_wrist_conf']>0.3]['left_wrist_y'], alpha = 0.2, color = 'r', label = 'kicks')
    
    plt.xlabel('X Co-ordinate', size=14)
    plt.ylabel('Y Co-ordinate', size=14)
    plt.title('Left Wrist Co-ordinates in Both Classes (Confidence > 0.2)')
    plt.legend(loc='upper right')
    
    
    fontsize=8
    labelsize=5
    ax1.set_title(str(option) + ' vs WnvPresent', fontsize=fontsize)
    ax1.set_ylabel(str(option),fontsize=fontsize)
    ax2.set_ylabel('WnvPresent',fontsize=fontsize)
    ax1.set_xlabel('Month',fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    
    st.pyplot(fig)
    
    
    # 
     
    


    # Weather patterns in Chicago
    st.header('Patterns Observed')
    st.subheader('Noticeable correlation between humidity, heat, and WNV+ mosquitoes')
  
    # text explaination for  
    st.write('A consistent positive correlation is observed between temperature, humidity and the presence of WNV positive mosquitoes, indicating a higher prevalence of WnvPresent during periods of increased temperature. This is consistent with existing literature. According to Lebl et al. (2013) the observed relationship can be attributed to the temperature-dependent development rates of mosquito life stages, including eggs, larvae, pupae, and adult survival rates. Similarly, according to Drakou et al. (2020), Studies have indicated that high humidity contributes to increased egg production, larval indices, mosquito activity, and overall influences their behavior. Furthermore, research has indicated that an optimal range of humidity, typically between 44% and 69%, stimulates mosquito flight activity, with the most suitable conditions observed at around 65%.')

    st.text("") # add extra line in between
    
    st.header('Mosquito Species As WNV Vectors')
    
    
    # now to create a table for the mosquito information
    species_with_virus = eda_df.pivot_table(values=['NumMosquitos'], index='Species',
                                       columns='WnvPresent', aggfunc='sum')

    # Calculate the overall total number of mosquitos across all species
    overall_total = species_with_virus['NumMosquitos'].sum().sum()
    
    # Calculate the overall total number of mosquitos with virus across all species
    overall_virus = species_with_virus['NumMosquitos',1].sum()
    
    # Create a new column for the percentage
    species_with_virus['Percentage_of_overall'] = species_with_virus[('NumMosquitos', 1)] / overall_total * 100
    species_with_virus['Percentage_of_virus'] = species_with_virus[('NumMosquitos', 1)] / overall_virus * 100
    
    cols = ['orange','paleturquoise']
    
    moz = species_with_virus.plot(kind='barh', stacked=True, figsize=(12,5), color = cols).figure
    plt.title("Number of Mosquitos with and without Virus")
    plt.xlabel("Number of Mosquitos")
    plt.legend(labels=["Without Virus", "With Virus"]);
    
    # visualise the mosquito species bar chart
    st.pyplot(moz)
    

    moz2 = plt.figure(figsize = (12,5))
    sns.barplot(data=eda_df,x='WnvPresent',y='Species', color = 'paleturquoise')
    plt.title("Probability of WNV Being Present By Species")
    
    st.pyplot(moz2)
    
    st.subheader('Culex Pipiens is the most likely vector for WNV')
    st.write("Based on the aforementioned observations, it can be inferred that the mosquito species Culex Pipiens and Culex Restuans are carriers of the West Nile virus, while the remaining species caught do not pose a risk.")
    st.write('Spraying efforts should therefore focus more on areas that have higher incidence of Culex Pipiens and Culex Restuans.')


if selected == 'Live Muay Th.AI':
    # title
    st.title('Live AI Training')
    st.subheader('by Wynne Chen')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)

    st.header('Chicago')
    
    # explain the animation
    st.write('Pick a date on the slider under the map to see the number of WNV positive mosquitoes, or press play to see the changing values over time.')
    
    # time to make the mosquito dataframe for mapping
    
    eda_df['Date'] = pd.to_datetime(eda_df['Date'])
    eda_df['Year-Month'] = eda_df['Date'].dt.strftime('%Y %m')
    eda_df['Year-Month'] = pd.to_datetime(eda_df['Year-Month'], format='%Y %m').dt.to_period('M')
    
    # Calculate total 'NumMosquitos'
    total_mosquito = eda_df.groupby(['Address','Year-Month'], as_index=False)['NumMosquitos'].sum()
    total_mosquito.sort_values(by='Year-Month', inplace=True)
    
    # Calculate median 'latitude' and 'longitude' for each address
    areas = eda_df.groupby('Address', as_index=False)[['Latitude', 'Longitude']].median()
    
    # Calculate total number of 'WnvPresent'
    virus = eda_df.groupby('Address', as_index=False)['WnvPresent'].sum()
    
    # merge datasets together
    mos_data = pd.merge(total_mosquito, areas, on='Address')
    mos_data = pd.merge(mos_data, virus, on='Address')
    
    # since we no longer need 'Address', drop col
    mos_data.drop('Address', axis = 1, inplace = True)
    
    # sort by 'Year-Month'
    mos_data.sort_values(by='Year-Month', inplace=True)
    
    
    # Convert dataframe to geodataframe
    mos_geo = gpd.GeoDataFrame(mos_data, geometry= gpd.points_from_xy(mos_data.Longitude, mos_data.Latitude))
    
    # Output with community areas added to the mosquito dataframe
    mos_chicago = gpd.sjoin(mos_geo, chicago, op='within')
    
    # Summary with some actionable content
    # community_infections is the df we will use for display because it is the cleanest summary
    # it will be called later with an option to choose how many rows you want to see
    community_infections = mos_chicago[['community','NumMosquitos', 'WnvPresent']].groupby('community').sum()
    community_infections.sort_values('WnvPresent', inplace = True, ascending = False)
    chicago_ltd = chicago[['community', 'geometry']]
    community_infections2 = chicago_ltd.merge(community_infections, on='community')
    community_infections2.sort_values('WnvPresent', inplace = True, ascending = False)
    community_infections2.reset_index(inplace = True)
    
    
    # Now for the actual animated map
    
    # Set the Mapbox access token
    px.set_mapbox_access_token('pk.eyJ1IjoiZ2l0aHViYmVyc3QiLCJhIjoiY2xqb3RtcjlwMWp4aDNscWNjdHZuNmU1ayJ9.BizJFoOXaa2H5jsYDkFeSg')
    
    
    # Create a scatter mapbox
    fig_m = px.scatter_mapbox(mos_chicago, 
                            lat=mos_chicago.geometry.y, 
                            lon=mos_chicago.geometry.x,
                            color='NumMosquitos', size='WnvPresent',
                            color_continuous_scale=px.colors.sequential.Jet,
                            hover_data=['NumMosquitos', 'WnvPresent', 'community'], zoom=9, animation_frame='Year-Month')
    
    
    # Create a layer for the community area boundaries
    layer_chicago = dict(
        sourcetype = 'geojson',
        source = chicago_geojson,
        type='fill',
        color='hsla(0, 100%, 90%, 0.2)',  
        below='traces',
        )
    
    # Add the community area boundaries layer to the scatter map
    fig_m.update_layout(mapbox_layers=[layer_chicago])
    
    # Update the layout
    fig_m.update_layout(mapbox_style= 'stamen-toner',
                      title='WNV+ vs. Mosquito count',
                      autosize=False,
                      width=1200,
                      height=1200,
                      )
    
    # Display the figure
    st.plotly_chart(fig_m)
    
    
    

    
    # Request input for how many rows to show
    header_n = st.slider('Select the top number of communities affected by positive WNV mosquitoes', 1, 20, 5)
    
    st.write('Note: Numbers shown below are cumulative. There are a total of 61 communities in our data set.')
    
    # Display the top n communities in table form
    show_communities = community_infections.head(header_n)
    st.dataframe(show_communities)
    
    # Display the top n communities in map form
    show_communities2 = community_infections2.head(header_n)
    
    m2 = fs.Map(location=[41.881832, -87.623177],tiles = 'Stamen Terrain', zoom_start=10, scrollWheelZoom=False)
    m2.choropleth(geo_data = chicago_geojson, 
                    data = show_communities2,
                    columns = ['community', 'WnvPresent'],
                    key_on = 'feature.properties.community',
                    fill_color = 'YlOrRd', 
                    fill_opacity = 0.7, 
                    line_opacity = 0.2,
                    legend_name = 'Number of WNV Positive Mosquitos per Neighbourhood in the years 2007, 2009, 2011, 2013')
        
    for lat, lon, community, wnv in zip(community_infections2.geometry.centroid.y, community_infections2.geometry.centroid.x , community_infections2.community, community_infections2.WnvPresent):
        fs.CircleMarker(location=[f'{lat}',f'{lon}'], radius = 1, color = 'green', fill = True, tooltip=f'{community}, Total Number of WNV Positive Mosquitoes: {wnv}').add_to(m2)    
    
    st_map = folium_static(m2, width=1200)


if selected == 'Upload A Video':
    # title
    st.title('Form Advice On Video')
    st.subheader('by Wynne Chen')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)
    
    st.header('Risk in your area')
    
    
    # dictionaries that are important
    
    # keypoint dictionary
    KEYPOINT_DICT = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
        }
    
    # colour dictionary
    
    KEYPOINT_EDGE_INDS_TO_COLOR = {
        (0, 1): 'm',
        (0, 2): 'c',
        (1, 3): 'm',
        (2, 4): 'c',
        (0, 5): 'm',
        (0, 6): 'c',
        (5, 7): 'm',
        (7, 9): 'm',
        (6, 8): 'c',
        (8, 10): 'c',
        (5, 6): 'y',
        (5, 11): 'm',
        (6, 12): 'c',
        (11, 12): 'y',
        (11, 13): 'm',
        (13, 15): 'm',
        (12, 14): 'c',
        (14, 16): 'c'
        }


    
    def update_df(df, features_to_fill, updated_features):
        for i, feature in enumerate(features_to_fill):
            df[feature] = updated_features[i]
        return df

    def model_predict(df, model=model):
        df2 = df.copy()
        df2['Trap'] = df2['Trap'].map(trap_map)
        df2.drop(['AddressNumberAndStreet','Latitude', 'Longitude'], axis=1, inplace=True)
        x = model.predict_proba(df2)[:,1]
        df['WnvProbability'] = x
        return df
    
    st.subheader('Fill in the following to see the risk of WNV in this time period.')
    species = st.selectbox('Species',['Culex Pipiens/Restuans', 'Culex Restuans', 'Culex Pipiens'], index=0)
    species = species_map[species]

    depart = st.slider('Departure from normal temperature', min_value=-20, max_value=20, step=1)

    sunrise = st.time_input('Sunrise')
    sunset = st.time_input('Sunset')
    timediff = dt.datetime.combine(dt.datetime.today(), sunset) - dt.datetime.combine(dt.datetime.today(), sunrise)
    timediff = 24 - timediff.seconds / 3600

    sunrise = sunrise.hour * 100 + sunrise.minute
    sunset = sunset.hour * 100 + sunset.minute

    codesum = st.multiselect('CodeSum', ['Normal', 'BR', 'HZ', 'RA', 'TS', 'VCTS'])
    codesum = 0.042585423329405826 # Codesum score for normal

    roll_sum_21_PrecipTotal = st.slider('Rolling Sum of Precipitation (inches) (21 days)', min_value=00, max_value=25, step=1)
    roll_sum_28_PrecipTotal = st.slider('Rolling Sum of Precipitation (inches) (28 days)', min_value=00, max_value=25, step=1)
    roll_mean_7_Tmin = st.slider('Minimum Temperature (°F) (7 days rolling mean)', min_value=40, max_value=90, step=1)
    roll_mean_28_Tmin = st.slider('Minimum Temperature (°F) (28 days rolling mean)', min_value=40, max_value=90, step=1)
    roll_mean_28_Tavg = st.slider('Average Temperature (°F) (28 days rolling mean)', min_value=40, max_value=90, step=1)
    
    date = st.date_input('Date')
    month = date.month
    year = date.year

    num_trap = st.number_input('Average number of times checked for each trap a day', min_value=0, step=1)
    roll_sum_14_num_trap = num_trap * 14
    speciesXroll_sum_28_num_trap =  species * (num_trap * 28)


    features_to_fill = ['Species', 'Depart', 'Sunrise', 'Sunset', 'CodeSum',
                        'roll_sum_21_PrecipTotal', 'roll_sum_28_PrecipTotal',
                        'roll_mean_7_Tmin', 'roll_mean_28_Tmin', 'roll_mean_28_Tavg',
                        'Month', 'Year', 'num_trap', 'roll_sum_14_num_trap',
                        'speciesXroll_sum_28_num_trap', 'timediff']
    if num_trap is not None:
        updated_features = [species, depart, sunrise, sunset, codesum,
                            roll_sum_21_PrecipTotal, roll_sum_28_PrecipTotal,
                            roll_mean_7_Tmin, roll_mean_28_Tmin, roll_mean_28_Tavg,
                            month, year, num_trap, roll_sum_14_num_trap,
                            speciesXroll_sum_28_num_trap, timediff]
        update_df(model_df, features_to_fill, updated_features)
        
        model_predict(model_df)

    st.subheader('Map')
    fig = px.scatter_mapbox(model_df,
                            lat='Latitude',
                            lon='Longitude',
                            hover_name='Trap',
                            hover_data=['AddressNumberAndStreet', 'Latitude', 'Longitude', 'WnvProbability'],
                            size='WnvProbability',
                            color='WnvProbability',
                            color_continuous_scale=[[0, 'rgb(255, 200, 200)'], [1, 'rgb(255, 0, 0)']],
                            range_color=[0, 1],
                            zoom=10)
    # Create a layer for the community area boundaries
    layer_chicago = dict(
    sourcetype = 'geojson',
    source = chicago_geojson,
    type='fill',
    color='hsla(0, 100%, 90%, 0.2)',  
    below='traces',
    )


    fig.update_layout(mapbox_layers=[layer_chicago],
                      mapbox_style='carto-positron',
                      margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                      height=700,
                      width=1000)

    st.plotly_chart(fig)
    
    st.subheader('Areas with the highest probability of WNV')
    high_proba = model_df[model_df['WnvProbability'] == model_df['WnvProbability'].max()][['AddressNumberAndStreet', 'WnvProbability']]
    high_proba.reset_index(inplace = True, drop = True)
    st.dataframe(high_proba, width = 600, height = 400)

    st.subheader('Areas with the lowest probability of WNV')
    low_proba = model_df[model_df['WnvProbability'] == model_df['WnvProbability'].min()][['AddressNumberAndStreet', 'WnvProbability']]
    low_proba.reset_index(inplace = True, drop = True)
    st.dataframe(low_proba, width = 600, height = 400)
    
