# Music-Recommendation-System
### Group members:
- [Irtaza Ahmed](https://github.com/irtazajawad) _(i221975@nu.edu.pk)_

## Introduction
This project provides a web-based interface for a music recommendation system that can play and recommend songs based on what the user is listening to. This is achieved by training an ML model which can then recommend songs by taking a single song as input. The model was trained on a sample of fma_large.zip file which contains 30s audio files. The dependencies used in the project are as follows:

## Dependencies:
- [Numpy]([https://docs.python.org/3/library/re.html](https://numpy.org/doc/))
- [SciKit Learn]([https://docs.python.org/3/library/csv.html](https://scikit-learn.org/0.21/documentation.html))
- [Py Spark]([https://docs.python.org/3/library/sys.html](https://spark.apache.org/docs/latest/api/python/index.html))
- [Liberosa]([https://docs.python.org/3/library/itertools.html](https://librosa.org/doc/))
- [Pandas](https://pandas.pydata.org/docs/)

## Dataset Used:
The data used for this project can be found at: https://scikit-learn.org/0.21/documentation.html

## Approach:
The following steps were implemented while working on this project
## Phase 1:
In this phase, the data was downloaded sampled, and then processed to extract features. These features were extracted using *Liberosa* and later normalized so that they could be uploaded to MongoDB efficiently. After the features were computed, the attributes of individual tracks were attached to the features using the CSV files provided in the fma_metadat.zip file, using the *utilities* module provided by the authors of the FMA repository. The features along with the attributes were extracted to a CSV file and also uploaded to MongoDB for later training of the model.

## Phase 2:
In this phase, the data was loaded from MongoDB and used for training the model. Spark was used to load the data via the Spark-Mongo connector. K-Means from Scikit learn were used to train the model which is accurate up to 40%. After training you can input a Track ID and get five similar songs. This model was later used in the web application.

## Phase 3: 
For each document, the **Term Frequency (TF)** was calculated. Term frequency represents the frequency of occurrence of each term within a document. This information was stored in a dictionary, where each key represents a document and its corresponding value is another dictionary containing _term-frequency pairs_.


## References:
- Hadoop Documentation: https://hadoop.apache.org/docs/stable/hadoop-mapreduce-client/hadoop-mapreduce-client-core/MapReduceTutorial.html
- Cholissodin, Imam & Seruni, Diajeng & Zulqornain, Junda & Hanafi, Audi & Ghofur, Afwan & Alexander, Mikhael & Hasan, Muhammad. (2020). [Development of Big Data App for Classification based on Map Reduce of Naive Bayes with or without Web and Mobile Interface by RESTful API Using Hadoop and Spark](https://www.researchgate.net/publication/348110835_Development_of_Big_Data_App_for_Classification_based_on_Map_Reduce_of_Naive_Bayes_with_or_without_Web_and_Mobile_Interface_by_RESTful_API_Using_Hadoop_and_Spark). Journal of Information Technology and Computer Science. 
- Vector Space Model: https://towardsdatascience.com/lets-understand-the-vector-space-model-in-machine-learning-by-modelling-cars-b60a8df6684f
