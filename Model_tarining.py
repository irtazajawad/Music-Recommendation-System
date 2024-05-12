from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

# Create a SparkSession
spark = SparkSession.builder \
    .appName("MusicRecommendation") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .getOrCreate()

# Read data from MongoDB collection into a PySpark DataFrame
df = spark.read.format("com.mongodb.spark.sql.DefaultSource") \
    .option("uri", "mongodb://localhost:27017/FMA_Sample.Track Info") \
    .load()

# Select the relevant columns
selected_cols = ['id', 'MFCC Features']

# Drop rows with missing values
df = df.dropna(subset=selected_cols)

# Flatten the nested array of MFCC Features
flatten_udf = udf(lambda x: [item for sublist in x for item in sublist] if isinstance(x, list) else [x], ArrayType(FloatType()))
df = df.withColumn("flattened_features", flatten_udf("MFCC Features"))

# Convert the features to a NumPy array
features = np.array(df.select("flattened_features").collect())

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Set the number of clusters
k = 10

# Train the K-means model
kmeans = KMeans(n_clusters=k)
kmeans.fit(scaled_features)

# Get the cluster assignments for each feature vector
labels = kmeans.labels_

# Compute the centroids of the clusters
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Function to get similar songs based on a given song ID
def get_similar_songs(song_id, num_songs=5):
    song_index = df.filter(df.id == song_id).select("flattened_features").collect()[0][0]
    song_index_scaled = scaler.transform([song_index])

    # Find the closest centroid to the song
    closest_centroid_index = pairwise_distances_argmin_min(song_index_scaled, centroids)[0][0]

    # Find other songs in the same cluster
    song_indices = np.where(labels == closest_centroid_index)[0]
    similar_songs = df.filter(df.id.isin(song_indices)).filter(df.id != song_id) \
        .select("id") \
        .limit(num_songs) \
        .collect()
    return [song.id for song in similar_songs]

# Example usage: Get similar songs for song with ID 123
similar_songs = get_similar_songs(123)
print("Similar songs for song ID 123:")
print(similar_songs)