import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.title('Detector de biodiversidad')

# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#          'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache_data
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# # Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# # Load 10,000 rows of data into the dataframe.
# data = load_data(10000)
# # Notify the reader that the data was successfully loaded.
# data_load_state.text("Done! (using st.cache_data)")

# ## Show raw data
# st.subheader('Raw data')
# st.write(data)

# ## Show histogram
# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(
#     data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)


# ##Plot a map
# st.subheader('Map of all pickups')
# st.map(data)

# ## Use dynamic filter
# # hour_to_filter = 17
# hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
# st.subheader(f'Map of all pickups at {hour_to_filter}:00')
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# st.map(filtered_data)

# ## Use checkbox
# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)
    
    
# -------------------------------------------------- #
#              UPLOAD IMAGE                          #
# -------------------------------------------------- #

# Function to Read and Manupilate Images
st.subheader('Upload image')

def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])

# Checking the Format of the page
if uploadFile is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = load_image(uploadFile)
    st.image(img)
    st.write("Image Uploaded Successfully")
else:
    st.write("Make sure you image is in JPG/PNG Format.")
    
# -------------------------------------------------- #
#              TAKE PHOTO                            #
# -------------------------------------------------- #  
st.subheader('Take photo')

enable = st.checkbox("Enable camera")
picture = st.camera_input("Take a picture", disabled=not enable)

if picture:
    st.image(picture)
    

## read the image file buffer as bytes
# if picture is not None:
#     # To read image file buffer as bytes:
#     bytes_data = picture.getvalue()
#     # Check the type of bytes_data:
#     # Should output: <class 'bytes'>
#     st.write(type(bytes_data))