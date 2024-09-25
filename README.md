# Apple Quality Detection Project : Project Overview
* Created a deep learning project that tends to detect bad quality apples according to their natural characteristics using autoencoders technique with a recall score that equals to 34.73 % and an accuracy score of 54.5 %.
* Collected data to work with from kaggle (a dataset named `Apple Quality` within Datasets section).
* Cleaned data up and engineered features so they will help us in the prediction process like standardizing & normalizing numerical features.
* Built an autoencoder from scratch that consists of two parts (Encoder & Decoder) to feed it with our data and try to get the best performance ever from our model.

## Code and Resources used
<b>Python Version :</b> 3.9<br>
<b>Packages :</b> Numpy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow<br>
<b>For Web Framework Requirements :</b> `pip install -r requirements.txt`

## Data Collection
Collect data from kaggle website to get a dataset that has 4000 apples. With each apple, we got the following :
* Apple ID
* Apple Size
* Apple Weight
* How much sweet an apple is
* How much crunchy an apple is
* How much juicy (contains a lot of Juice) an apple is
* How much ripe (arriving at such a stage of growth or development as to be ready for reaping) an apple is
* How much acidic an apple is (Its Juice is acidic)
* Apple Quality (Whether it's good or bad)

## Data Cleaning
Clean the data up to make it ready for modeling process. I made the following changes:
* Encoded target variable and transform its data type from an object to be numerical using a simple mapping method
* Standardized float features using a standard scaler

## EDA with Data Visualization (Matplotlib & Seaborn)
I looked at the distributions of the data, calculated and visualized Pearson's correlation coefficient between all variables and explored the value counts for the target variable (which is the only categorical variable in the dataset) to extract insights and patterns that could be useful in the rest of the project. Below are some of the visualizations I made:<br><br><br>
<img src="size_dist.png"><br>
<img src="corr_mat.png"><br>
<img src="quality_pie.png">

## Model Building
For anomaly detection, we only need genuine class of the data. So, we split up our data into training and test sets with 20 % as a test size and divide genuine and abnormal classes in both training and test datasets. After that, we construct the autoencoder from scratch which is composed of two different parts (Encoder & Decoder) :
* The encoder typically consists of one or more fully connected layers that transform the input data into a lower-dimensional representation. The number of nodes in the hidden layer is typically smaller than the number of nodes in the input and output layers, which forces the network to learn a compressed representation of the input data. The activation function used in the encoder can be any non-linear function, such as a sigmoid or a rectified linear unit (ReLU), which allows the network to capture non-linear relationships in the input data.
* The decoder is typically a mirror image of the encoder, with one or more fully connected layers that transform the compressed representation back into the original input space. The output layer of the decoder should have the same number of nodes as the input layer, so that the decoder can produce a reconstruction of the input data. The activation function used in the decoder is typically the same as the one used in the encoder.<br>
After building our model, we compiled it using Adam as an optimizer and mean-absolute-error as a loss function and trained it using only only genuine class of the training set.

## Reconstruction & Classification
We will start by making a prediction on the test set which consists of both classes (genuine & abnormal). After this, we can define a threshold and a metric, depending upon the need. The idea is simple:

* If the Reconstruction error is lower than the threshold, the sample is good.
* If the Reconstruction error is higher than the threshold, the sample is bad.<br>
This is because the model was trained with samples of genuine class, so anything outside of this threshold is considered an anomaly.<br><br>Choosing the right threshold is crucial in anomaly detection with autoencoders because it determines the tradeoff between detecting anomalies and generating false positives. The threshold determines the cutoff point for the reconstruction error, above which a data point is classified as anomalous. If the threshold is set too low, the autoencoder will classify many normal data points as anomalies, resulting in a high false positive rate. On the other hand, if the threshold is set too high, the autoencoder may miss some true anomalies, resulting in a high false negative rate.<br><br>
The metric we will choose for this problem is Recall, as we want to reduce False Negatives. Any Fraud apple classified as Genuine may lead to unnoticed problems in the stomach, since there will never be anomaly detected. To find the right threshold value, several values will be tested to find the best combination of metrics. While our goal is to improve Recall, we will also keep track of the accuracy. The tested values will be percentiles of the reconstruction error values.

## Model Performance
Our autoencoder hasn't performed very bad on test set as we get the following as final results:
* <b>Recall score :</b> 34.73 %
* <b>Accuracy score :</b> 54.5 %
