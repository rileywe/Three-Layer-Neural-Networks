# Three-Layer-Neural-Networks

##### Author: 
Riley Estes

### Abstract
Exploration and comparison of the effectiveness of a 3 layer feed forward neural network in fitting to a curve and classifying MNIST digits. The neural network model in question is shown to perform poorly and overfit greatly in simple regression tasks in comparison to a linear fit, but will provide an effective and cheap method to classify image data. The performance of the model versus an LSTM, SVM, and Decision Tree is explored and the 3 layer Feed Forward NN comes in a close second place in terms of accuracy behind the SVM, yet takes only 3/4 the time (and thus computational cost) to get there. 

### Introduction and Overview
A 3 layer Feed Forward Neural Network is a semi-complex nonlinear machine learning model that can perform well in many data processing situations. In this program, the model is tested on a simple regression problem, as well as a more complex classification problem with the MNIST handwritten digits dataset of well framed standardized images. The accuracy of the model in comparison to alternative models will be shown here, and the computational cost and complexity will also be examined. 

### Theoretical Background

#### Neural Network
A Neural Network, also known as a Multi-Layer Perceptron (MLP) is a machine learning algorithm where data passes through a series of layers of nodes connected with each other (fully or partially) by weights. This creates a nonlinear and very complicated network because each node in a layer is generally connected to all the nodes in the next layer, each with its own weight. That means that each node in a layer is the sum of all of the nodes in the last layer multiplied by each connection's particular weight. These networks require training with a training set, and are then tested on a test set of data. In training, the values of all the weights are updated (using backpropagation) based on the incoming data (and its labels for supervised learning). The model can then be tested on the test data to see how well it processed the training data, and how well its weights are set to achieve the data processing task. Neural Networks often perform very well on complicated tasks, but require huge amounts of data to do so. 

#### Feed-Forward Neural Network
A Feed-Forward Neural Network is one that has a linear flow of data from the input to the output. That is, there are no loops or ways data can be repeated or looped in the network. This is the simplist neural network design.

#### Long Short-Term Memory Neural Network (LSTM)
An LSTM is a type of Neural Network that is designed to process sequential data. Similar to a Recurrent Neural Network, an LSTM creates feedback loops so that it can "remember" data and use previous data in order to process current data. In addition to this however, the LSTM implements a memory cell where it can selectively store and access data in these cells for later use when processing future information. It adds an extra layer of memory to the Recurrent Neural Network design to further increase its temporal processing abilities. 

#### Support Vector Machine (SVM)
Support Vector Machines attempt to find the best possible boundary between different classes in the data by identifying a hyperplane in a high-dimensional feature space that maximizes the margin (distance from the hyperplane to the nearest data point) between the classes instead of directly minimizing the error. SVMs are particularly useful when the number of dimensions in the feature space is large, and the data is not linearly separable. In such cases, SVMs can use a kernel function to transform the data into a higher-dimensional space, where it may be possible to separate the classes by a hyperplane. This model can very effectively classify data with supervised machine learning. However, it does so at great computational cost with a complexity of at least O(N^2).

#### Decision Tree
Decision trees are another type of classification algorithm that works by splitting the input data into smaller subsets based on the input features until the data cannot be further split, or a certain number of splits have occured. In each new split, the amount of data to consider has been reduced, but there are more data splits that need to be computed as the tree splits and separates further. This method produces an easy to visualize tree-like model where each internal node corresponds to a decision rule based on one or more input features, and each leaf node corresponds to a prediction for the target variable. 

### Algorithm Implementation and Development
The first segment of this program uses the following data:
```
X_nums=np.arange(0,31)
Y_nums=np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```
Then the 3 layer Feed Forward Neural Network is instantiated with a size 9 hidden layer, relu activation after the hidden layer, adam solver, and 10000 max iterations:
```
three_layer_NN = MLPRegressor(hidden_layer_sizes=(9,), activation='relu', solver='adam', max_iter=10000)
```
The function
```
def modelErrors (model, X_train, X_test, Y_train, Y_test):
```
will print out the Mean Squared Errors for the given model on the training and testing data, as well as the Least Square Errors for each trainign and testing point.

The model is first fit to a training set of the first 20 points in the data, and secondly to a training set of the first and last 10 points of the data. The model is also fit to the test data (middle 11 points) in the second split here:
```
three_layer_NN.fit(X_test.reshape(-1, 1), Y_test)
```
The errors are printed in all 3 cases. 

Switching over to the MNIST data, a PCA analysis is taken at first to reduce computational cost:
```
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_mnist)
```
and the data is randomly split into 80%/20% training/testing
```
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y_mnist, test_size=0.2, random_state=0)
```

The 3 layer Feed-Forward Neural Network design is applied to the MNIST data, this time with a size 100 hidden layer:
```
three_layer_mnist = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=0)
```
Then, the 3 layer Feed-Forward Neural Network is compared to 3 other methods for the MNIST data:
#### LSTM
The LSTM is also 3 layers but uses a size 128 hidden layer instead with a hyperbolic tangent (default) activation function. It also uses the adam optimizer, with cross-entropy loss and softmax output activation. 
```
model_lstm = Sequential()
model_lstm.add(LSTM(128, input_shape=(X_train.shape[1], 1)))
model_lstm.add(Dense(10, activation='softmax'))
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
#### SVM
The SVM model is the simple Support Vector Classifier from sklearn shown here:
```
model_svm = SVC()
model_svm.fit(X_train, Y_train)
```
The complexity greatly reduces as we step away from neural nets and back towards classical machine learning
#### Decision Tree
Like the SVM, the Decision Tree approach also uses the sklearn package:
```
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, Y_train)
```

### Computational Results
In the first train/test split for the first dataset, the training MSE for the 3 layer Feed Forward Neural Network is 5.03, and the test MSE is 11.15. Notice the sizeable gap between the errors. There is a large amount of overfitting in this model to the dataset. To address this, dropout could be used to keep the model from fitting to the training data too much and too fast. Additionally, most LSEs for each point in the training set are roughly between 0 and 5, but a few are above 10 and 1 above 20. The test set has even more extreme numbers and less consistency. Overall, there is a moderate to high amount of variance in the LSEs of each point, making the model very unpredicatble and a sign of poor training. 
In the second train/test split where the training data is better dispersed throughout the whole dataset, the model performed notably better. The training MSE is 3.64 and the test MSE is 8.97. Although there is an improvement that is apparent with the increase in data distribution quality, there is still a large amount of overfitting, and the same inconsistencies in LSEs can be seen (although the extreme values are far less extreme, and overall cover a much smaller range). Refitting the model to the test data instead of the training data yields very similar training and test MSEs with similar variances. The training MSE (not what the model was fit on) is 3.30, and the test MSE (what the model was fit on) is 8.77. Uniquely, even with half the amount of fitting data, the model performed slightly better when the fitting data is in the middle of the dataset. 
To compare the neural network here to the models fit to the same data in the Regression for EE399 repository:
These models both perform worse on the data set than the simpler linear and sinusoidal regression models. For the first dataset where the first 20 points are used for training, the test MSE is 11.15 for the neural network, and 3.53 for the linear model. For the second dataset where the first and last 10 points are used for training, the test MSE is 8.97 for the neural network, and 2.95 for the linear model. When fitted to the test data instead, the training MSE for the neural network is 3.30, which is comparable to the linear fit, except that the test MSE is still very bad. The linear regression model performs much better in both cases than the neural network at a fraction of the computational cost, making it the obvious choice for simple curve-fitting tasks such as these. 

For the MNIST data transformed to the first 20 PCA components, the 3 layer Feed Forward Neural Network performed much better and is much more suited for image classification. The accuracy for the model on the MNIST data is 95.69%.
In comparison, the LSTM performed well but not nearly as well as the first method. The accuracy for the LSTM on the PCA MNIST data is 86.56%. This makes intutive sense because the LSTM is designed to process sequential data with a temporal element, but each image in the MNIST dataset has no relationship with the images next to it, and has no temporal component. However, this accuracy is still pretty good for a model not well suited for the task. 
The SVM performed the best overall with an accuracy of 97.30%. Image classification in such a standardized dataset is a rather simple task, so the simple SVM performes really well because it can effectively find the hyperplanes needed to classify the digits from the uniformly formatted images. 
The Decision tree did not perform very well with only 84.41% accuracy. Some of the digits are written in weird and inconsistent ways, so its harder to create a rigid decision tree based on characteristics that may be different across all of the images of the same digit. 
The 3 layer neural network was able to achieve its accuracy in 27.6 seconds as compared to the SVMs 39.0 seconds, making it a quicker and potentially more efficient choice for large datasets. 

### Summary and Conclusions
Overall, the 3 layer Fedd Forward Neural Network does not perform well in simple curve-fitting tasks, but does considerably well, expecially for its simplicity, on standardized classification tasks such as with the MNIST digits. For regression, even a simple linear fit has the neural network beat by a large margin because the network suffers heavily from overfitting. It's simply too complex for such a simple task, and will end up "overthinking" it. On the other hand, the simple neural network model performed very well on classifying MNIST digits, achieving a close second place behind the SVM algorithm. Notably, the neural network performed better than the more complex, temporal-thinking LSTM, and the Decision Tree models. The neural network does however compute faster than the SVM, and may be the most efficient classification method tested here. 
In conclusion, for simple regression tasks, use a linear or low-degree polynomial fit. For classification tasks, it will be better to use a Feed Forward Neural Network, or an SVM if computational efficiency can be sacrificed for a small goin in accuracy.


