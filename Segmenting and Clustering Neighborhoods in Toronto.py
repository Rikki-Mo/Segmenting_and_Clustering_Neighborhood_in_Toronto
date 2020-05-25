#!/usr/bin/env python
# coding: utf-8

# # Segmenting and Clustering Neighborhoods in Toronto

# In this assignment, we are required to explore and cluster the neighborhoods in Toronto. The Notebook is build using a Wikipedia page in order to obtain the data that is in the table of postal codes and to transform the data into a pandas dataframe 

# ## Question 1

# ### 1.1 Creating the Notebook.

# Notebook is created - Segmenting and Clustering Neighborhoods in Toronto

# ### 1.2 Scrapping the wikipedia Page

# In[1]:


#importing the libraries
from bs4 import BeautifulSoup
import lxml       
import requests
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library


# In[2]:


#Scrapping the website
source = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M").text
soup = BeautifulSoup(source, 'lxml')

table = soup.find("table")
table_rows = table.tbody.find_all("tr")

res = []
for tr in table_rows:
    td = tr.find_all("td")
    row = [tr.text for tr in td]
    res.append(row)


# ### 1.3 Creating the DataFrame

# In[3]:


# headers="Postcode,Borough,Neighbourhood"
df = pd.DataFrame(res, columns = ["PostalCode", "Borough", "Neighborhood"])

#Remove "\n" at the end of each string in the PostalCode, Borough, Neighborhood column
df["PostalCode"] = df["PostalCode"].str.replace("\n","")
df["Borough"] = df["Borough"].str.replace("\n","")
df["Neighborhood"] = df["Neighborhood"].str.replace("\n","")
df.head()


# In[4]:


# Only processing the cells that have an assigned borough. 
# Ignoring the cells with a borough that is Not assigned

indexNum = df[df['Borough'] == 'Not assigned'].index
df.drop(indexNum, inplace = True)
df.head()


# In[5]:


# If a cell has a borough but a Not assigned neighborhood
# then the neighborhood will be the same as the borough

df.loc[df['Neighborhood'] == 'Not assigned', 'Neighborhood'] = df['Borough']
df.head()


# In[6]:


#Combining rows with the same postal code area
df_group = df.groupby(['PostalCode', 'Borough'], sort = False).agg( ','.join)
df_new = df_group.reset_index()
df_new.head()


# In[7]:


df_new.shape


# ## Question 2

# ### 2.1 Get the latitude and longitude coordinates of a given postal code

# In[8]:


df_geo_coor = pd.read_csv("http://cocl.us/Geospatial_data")
df_geo_coor.head()


# In[9]:


df_toronto = pd.merge(df_new, df_geo_coor, how='left', left_on = 'PostalCode', right_on = 'Postal Code')
# remove the "Postal Code" column
df_toronto.drop("Postal Code", axis=1, inplace=True)
df_toronto.head()


# ## Question 3

# ### Explore and cluster the neighborhoods in Toronto

# ### 3.1 Get the latitude and longitude values of Toronto

# In[10]:


address = "Toronto, ON"

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto city are {}, {}.'.format(latitude, longitude))


# ### 3.1.1 Create a map of the whole Toronto City with neighborhoods superimposed on top and adding markers

# In[35]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, Neighbourhood in zip(df_toronto['Latitude'], df_toronto['Longitude'], df_toronto['Borough'], df_toronto['Neighborhood']):
    label = '{}, {}'.format(Neighbourhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)
map_toronto


# ###  3.1.2 Map of a part of Toronto City

# We are going to work with only the boroughs that contain the word "Toronto".

# In[36]:


# "denc" = [D]owntown Toronto, [E]ast Toronto, [N]orth Toronto, [C]entral Toronto
df_toronto_denc = df_toronto[df_toronto['Borough'].str.contains("Toronto")].reset_index(drop=True)
df_toronto_denc.head()


# Plot again the map and the markers for this region.

# In[37]:


map_toronto_denc = folium.Map(location=[latitude, longitude], zoom_start=12)
for lat, lng, borough, neighborhood in zip(
        df_toronto_denc['Latitude'], 
        df_toronto_denc['Longitude'], 
        df_toronto_denc['Borough'], 
        df_toronto_denc['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto_denc)  

map_toronto_denc


# ### 3.2 Define Foursquare Credentials and Version

# In[38]:


CLIENT_ID = '4WKTWAPUF2OQJ0NHQNIWZ1WD0I0S55KDUN1YV2EJRML5NKZO' # your Foursquare ID
CLIENT_SECRET = '1PJUNQC2D2NAGHOYXKQZJP55J0HFWJQBJAOHGM1ATLUW4FSI' # your Foursquare Secret
VERSION = '20200525' # Foursquare API version


# ### 3.2.1 Explore the first neighborhood in our data frame "df_toronto"

# In[39]:


neighborhood_name = df_toronto_denc.loc[0, 'Neighborhood']
print(f"The first neighborhood's name is '{neighborhood_name}'.")


# Get the neighborhood's latitude and longitude values.

# In[40]:


neighborhood_latitude = df_toronto_denc.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = df_toronto_denc.loc[0, 'Longitude'] # neighborhood longitude value

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# ### 3.2.2 Now, let's get the top 100 venues that are in The Beaches within a radius of 500 meter

# Function that extracts the category of the venue

# In[41]:


LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)

# get the result to a json file
results = requests.get(url).json()


# Now we are ready to clean the json and structure it into a pandas dataframe

# In[42]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# ### 3.2.3 Explore neighborhoods in a part of Toronto City

# We are working on the data frame df_toronto_denc. Recall that, this region contain DENC of Toronto where,
# 
# "DENC" = [D]owntown Toronto, [E]ast Toronto, [N]orth Toronto, [C]entral Toronto
# 
# First, let's create a function to repeat the same process to all the neighborhoods in DENC of Toronto

# In[43]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    venues_list=[]
    
    for name, lat, lng in zip(names, latitudes, longitudes):
        # print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# Now write the code to run the above function on each neighborhood and create a new dataframe called toronto_denc_venues

# In[44]:


toronto_denc_venues = getNearbyVenues(names=df_toronto_denc['Neighborhood'],
                                   latitudes=df_toronto_denc['Latitude'],
                                   longitudes=df_toronto_denc['Longitude']
                                  )


# In[45]:


toronto_denc_venues.head()


# Let's check how many venues were returned for each neighborhood.

# In[46]:


toronto_denc_venues.groupby('Neighborhood').count()


# #### Let's find out how many unique categories can be curated from all the returned venues

# In[47]:


print('There are {} uniques categories.'.format(len(toronto_denc_venues['Venue Category'].unique())))


# ### 3.2.4 Analyze Each Neighborhood

# In[48]:


toronto_denc_onehot = pd.get_dummies(toronto_denc_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_denc_onehot['Neighborhood'] = toronto_denc_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_denc_onehot.columns[-1]] + list(toronto_denc_onehot.columns[:-1])
toronto_denc_onehot = toronto_denc_onehot[fixed_columns]

toronto_denc_onehot.head()


# Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[49]:


toronto_denc_grouped = toronto_denc_onehot.groupby('Neighborhood').mean().reset_index()
toronto_denc_grouped.head()


# #### Check the 10 most common venues in each neighborhood.

# In[50]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    return row_categories_sorted.index.values[0:num_top_venues]

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_denc_grouped['Neighborhood']

for ind in np.arange(toronto_denc_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_denc_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ### 3.2.5 Cluster neighborhoods

# Run k-means to cluster the neighborhood into 5 clusters.

# In[51]:


# set number of clusters
kclusters = 5

toronto_denc_grouped_clustering = toronto_denc_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_denc_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood

# In[52]:


# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

toronto_denc_merged = df_toronto_denc

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_denc_merged = toronto_denc_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

toronto_denc_merged.head() # check the last columns!


# Finally, let's visualize the resulting clusters

# In[53]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(
        toronto_denc_merged['Latitude'], 
        toronto_denc_merged['Longitude'], 
        toronto_denc_merged['Neighborhood'], 
        toronto_denc_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ### 3.2.6 Examine Clusters

# Now, you can examine each cluster and determine the discriminating venue categories that distinguish each cluster.

# #### Cluster 1

# In[54]:


toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 0, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]


# #### Cluster 2

# In[55]:


toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 1, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]


# #### Cluster 3

# In[56]:


toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 2, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]


# #### Cluster 4

# In[57]:


toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 3, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]


# #### Cluster 5

# In[58]:


toronto_denc_merged.loc[toronto_denc_merged['Cluster Labels'] == 4, toronto_denc_merged.columns[[1] + list(range(5, toronto_denc_merged.shape[1]))]]

