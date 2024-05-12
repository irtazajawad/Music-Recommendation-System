# Music-Recommendation-System
### Group members:
- [Irtaza Ahmed](https://github.com/irtazajawad) _(i221975@nu.edu.pk)_

## Introduction
This project provides a web-based interface for a music recommendation system that can play and recommend songs based on what the user is listening to. This is achieved by training an ML model which can then recommend songs by taking a single song as input. The model was trained on a sample of fma_large.zip file which contains 30s audio files. The dependencies used in the project are as follows:

## Dependencies:
- Numpy
- [SciKit Learn]([https://docs.python.org/3/library/csv.html](https://scikit-learn.org/0.21/documentation.html))
- [Py Spark]([https://docs.python.org/3/library/sys.html](https://spark.apache.org/docs/latest/api/python/index.html))
- [Liberosa]([https://docs.python.org/3/library/itertools.html](https://librosa.org/doc/))
- [Pandas](https://pandas.pydata.org/docs/)
- OS

## Dataset Used:
The data used for this project can be found at: https://scikit-learn.org/0.21/documentation.html

## Approach:
The following steps were implemented while working on this project
## Phase 1:
In this phase, the data was downloaded sampled, and then processed to extract features. These features were extracted using *Liberosa* and later normalized so that they could be uploaded to MongoDB efficiently. After the features were computed, the attributes of individual tracks were attached to the features using the CSV files provided in the fma_metadat.zip file, using the *utilities* module provided by the authors of the FMA repository. Another model was also trained using SparkML but due to complications, it was dropped in favor of Scikit Learb. The features along with the attributes were extracted to a CSV file and also uploaded to MongoDB for later training of the model.

## Phase 2:
In this phase, the data was loaded from MongoDB and used for training the model. Spark was used to load the data via the Spark-Mongo connector. K-Means from Scikit Learn were used to train the model which is accurate up to 40%. After training you can input a Track ID and get five similar songs. This model was later used in the web application.

## Phase 3: 
In this phase, the code loads a trained machine learning model, specifically a KMeans clustering model, and a StandardScaler model. These models are essential for generating song recommendations. The code also retrieves the cluster centroids from the KMeans model. The accompanying HTML files from templates are used to mark up the website.

## References:
- MFCC Features: (https://medium.com/@derutycsl/intuitive-understanding-of-mfccs-836d36a1f779)
- K Means(https://en.wikipedia.org/wiki/K-means_clustering)
- Mongo DB connectors: https://www.mongodb.com/products/integrations/connectors
- 
