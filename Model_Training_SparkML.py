from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

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

# Define a UDF to transform the MFCC Features array into a Vector
transform_features_udf = udf(lambda x: Vectors.dense(x), VectorUDT())

# Create a new DataFrame with transformed features
df = df.withColumn("features", transform_features_udf("MFCC Features"))

# Create a vector assembler to combine the features into a single vector column
vector_assembler = VectorAssembler(inputCols=['features'], outputCol='feature_vector')
df = vector_assembler.transform(df)

# Scale the features
scaler = StandardScaler(inputCol='feature_vector', outputCol='scaled_features')
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

# Set the number of clusters
k = 10

# Train the K-means model
kmeans = KMeans(featuresCol='scaled_features', k=k)
model = kmeans.fit(df)

# Define a UDF to compute the cluster assignment for a given feature vector
predict_cluster_udf = udf(lambda x: model.predict(x), StringType())

# Create a new DataFrame with the predicted cluster assignments
df = df.withColumn("cluster", predict_cluster_udf("scaled_features"))

# Function to get similar songs based on a given song ID
def get_similar_songs(song_id, num_songs=5):
    song_cluster = df.filter(df.id == song_id).select("cluster").collect()[0][0]
    similar_songs = df.filter(df.cluster == song_cluster).filter(df.id != song_id) \
        .select("id", "cluster") \
        .limit(num_songs) \
        .collect()
    return [song.id for song in similar_songs]

# Example usage: Get similar songs for song with ID 123
similar_songs = get_similar_songs(123)
print("Similar songs for song ID 123:")
print(similar_songs)