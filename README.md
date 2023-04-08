# <ins>Deep Machine Learning Challenge</ins>
Module 21 Neural Networks and deep learning

## Background

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With my knowledge of machine learning and neural networks, I’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years.

## Preprocess the Data

Using my knowledge of Pandas and scikit-learn’s StandardScaler(), I preprocessed the dataset. The charity_data.csv was read to a Pandas DataFrame, using Google Colab notebook. Once this was done, I identified the target and feature variables for the model. In my first attempt, I dropped the EIN and NAME columns, and determined the number of unique values for each column. These were used to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful. `pd.get_dummies()` was used to encode categorical variables. I then split the preprocessed data into a features array, X, and a target array, y. Using these arrays and the train_test_split function, I split the data into training and testing datasets.


## Compile, Train, and Evaluate the Model

Using my knowledge of TensorFlow and Keras, I designed a neural network model to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. Once I completed this step, I compiled, trained, and evaluated my binary classification model to calculate the model’s loss and accuracy.

* First attempt

![first_attempt](https://user-images.githubusercontent.com/116304118/230739546-9a2eb2fb-a366-4227-bfc1-7dc870af502d.png)


![first_attempt1](https://user-images.githubusercontent.com/116304118/230739532-2bdd70d1-d104-4f32-bff5-a945c1e1de49.png)


To view the HDF5 file, click here [`AlphabetSoupCharity.h5`]("https://github.com/HJandu/deep-learning-challenge/blob/main/h5_files/AlphabetSoupCharity.h5")

## Optimize the Model


Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

Create the first hidden layer and choose an appropriate activation function.

If necessary, add a second hidden layer with an appropriate activation function.

Create an output layer with an appropriate activation function.

Check the structure of the model.

Compile and train the model.

Create a callback that saves the model's weights every five epochs.

Evaluate the model using the test data to determine the loss and accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.
