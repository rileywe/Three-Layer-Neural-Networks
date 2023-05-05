# Three-Layer-Neural-Networks


##### Author: 
Riley Estes

### Abstract


### Introduction and Overview
The MNIST data set is a easy to visualize dataset that can still prove a challenge for more rudimentary machine learning algorithms. There is a lot of opportunity to analyse the MNIST data, 

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



### Computational Results


### Summary and Conclusions
