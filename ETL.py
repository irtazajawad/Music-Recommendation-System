import pandas as pd
import os
import librosa
import numpy as np
import utilities
import pickle
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# Read the CSV file containing track metadata
metadata_df = utilities.load('/home/irtaza/Downloads/Data Analysis/fma_metadata/tracks.csv')
audio_directory = '/home/irtaza/Downloads/Data Analysis/fma_large/000'
# Create a list to store the tuples of features and track metadata
feature_metadata_list = []

# Extract MFCC features
for filename in os.listdir(audio_directory):
    if filename.endswith('.mp3'):
        audio_filepath = os.path.join(audio_directory, filename)
        # Load the audio data using Librosa
        audio_data, sample_rate = librosa.load(audio_filepath)
        # Extract MFCC features
        mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
        # Normalize the MFCC features
        scaler = StandardScaler()
        mfcc_features_normalized = scaler.fit_transform(mfcc_features)
        # Reshape the features to 1D array
        mfcc_features_1d = mfcc_features_normalized.ravel()
        # Get the track metadata from the CSV based on the filename
        index = int(filename[:-4])  # Convert filename to integer index
        if index in metadata_df.index:
            track_info = metadata_df.loc[index]
            # Store the track metadata along with the features as a tuple
            feature_metadata_list.append((index, track_info.loc[('track', 'title')], track_info.loc[('artist', 'name')], track_info.loc[('track', 'genre_top')], mfcc_features_1d))


# Converting to DataFrame
df = pd.DataFrame(feature_metadata_list, columns=['id', 'Track Name', 'Artist', 'Genre', 'MFCC Features'])
df.set_index('id', inplace=True)

# df.to_csv('track_metadata_with_features.csv')



# Converting the tuple list to dict list for mongo upload
feature_list_for_upload = []
for index, title, artist, genre, features in feature_metadata_list:
    feature_metadata_dict = {
        'id' : index,
        'Track Name' : title,
        'Artist' : artist,
        'Genre' : genre,
        'MFCC Features': features.tolist()
    }
    feature_list_for_upload.append(feature_metadata_dict)



# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['FMA_Sample']
collection = db['Track Info']

# Insert the documents into the MongoDB collection
collection.insert_many(feature_list_for_upload)

# Retrieve the documents from the MongoDB collection
result_all = collection.find()


# Print the documents
for document in result_all:
    print("id:", document['id'])
    print("Track Name:", document['Track Name'])
    print("Artist:", document['Artist'])
    print("Genre:", document['Genre'])
    print("MFCC Features:")
    print(mfcc_features)
    print()


# for data in feature_list_for_upload:
#     print("Track ID: ", data['id'])
#     print("Track Name:", data['Track Name'])
#     print("Artist:", data['Artist'])
#     print("Genre:", data['Genre'])
#     print("MFCC features:")
#     print(pickle.loads(data['MFCC Features']))
#     print("\n")
