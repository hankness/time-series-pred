import pandas as pd
import numpy as np

# Class which handles all of the perceptron magic
class Perceptron():
    """ # Perceptron
    A class containing code for a single layer perceptron.

    Args:
        training_method (string): The kind of training we wish to perform.
    """

    
    def __init__(self, training_method, weights=None, eta=None):
        self.selected_method = training_method
        # generate weights if not provided
        if weights is None:
            self.weights = np.random.normal(size=(3), loc=0, scale=1)
        else:
            self.weights = weights

        # generate eta if not provided
        if eta is None:
            self.eta = 0.001
        else:
            self.eta = eta


    # This is the actual function that trains the model as per instructions
    def train_model(self, dataset):
        """ # train_model
        Trains the model using the training method provided. 

        Args:
            dataset (pd.DataFrame): The raw dataset to train the model on.

        Return: 
            - List of weights after each epoch of training
            - List of accuracies after each epoch of training
            - List of mean squared erros after each epoch of training
        """

        # Get the features and labels
        X = dataset[["x1", "x2"]]   # features (called patterns in assignment)
        y = dataset["y"]            # labels (called targets in assignment)

        # As per lab instructions (page 3), add an extra column to X filled with ones
        num_rows = X.shape[0]
        column_with_ones = np.ones((num_rows, 1))
        X = np.hstack((X, column_with_ones))
        
        # Initialize weights from the normal distribution
        num_epochs = 1000
        #bias = np.random.normal(size=(1, num_rows), loc=0, scale=1)         # Double check that this is the correct initialization of the values

        # Map training methods to functions and get the function from the selected method
        training_functions_map = {
            "PERCEPTRON": self._train_perceptron,
            "DELTA_RULE_SEQUENTIAL": self._train_delta_rule_sequential,
            "DELTA_RULE_BATCH": self._train_delta_rule_batch,
        }
        training_method = training_functions_map[self.selected_method]

        accuracy_per_epoch_list = list() 
        mean_squared_error_per_epoch_list = list()
        
        weights_list = []
        
        
        # Actual training
        for epoch in range(num_epochs):
            self.weights = training_method(X, y, self.weights)
            weights_list.append(np.copy(self.weights))
            # weights[epoch] = training_method(X, y, weights[epoch-1])
            
            # print(weights_list[epoch].shape)  # 1 x 3
            # print(X.shape)  # 200 x 3
            
            y_hat = weights_list[epoch] @ X.T
            
            # calculate model mean square error
            squared_errors_list = (y - y_hat) ** 2    # assuming y, predictions are numpy arrays
            MSE = squared_errors_list.mean()
            mean_squared_error_per_epoch_list.append(MSE)

            # generate model predictions
            predictions = self._step(y_hat)

            # calculate model accuracy
            correct_predictions = np.sum(y == predictions)
            accuracy = correct_predictions / len(y)
            accuracy_per_epoch_list.append(accuracy)
            


        return weights_list, accuracy_per_epoch_list, mean_squared_error_per_epoch_list

    # Private functions for training the perceptron
    def _train_perceptron(self, X, y, W):
        """ # _train_perceptron
        Train the Perceptron using the basic training model: W * X + b

        """

        for i in range(len(X)):
            y_pred = self._step(W @ X[i])
            target = y[i]

            # check if the weights need to be adjusted
            if y[i] != y_pred:
                W += self.eta * (target - y_pred) * X[i]
        return W
    



    def _train_delta_rule_sequential(self, X, y, W):
        """ # _train_delta_rule_sequential
        Train the Perceptron using the delta rule sequentially: -eta*(W*X - y)*X
        """

        for i in range(len(X)):
            y_pred = W @ X[i]
            target = y[i]
            W += self.eta * (target - y_pred) * X[i]
        
        return W


    def _train_delta_rule_batch(self, X, y, W):
        """
        Train the Perceptron using the delta rule in batch: -eta*(W*X - y)*X
        """
        # print(W.shape)  # 3 x 1
        # print(X.shape)  # 200 x 3
        # print(y.shape)  # 200 x 1
        
        # TODO make less transposes?
        y_pred = W.T @ X.T
        target = y.T
        W += self.eta * (target - y_pred) @ X
        
        return W

    def _step(self, y_hat):
        # Predict for single number
        if isinstance(y_hat, (int, float)):
            y_hat = 1 if y_hat > 0 else -1

        # Predict for array
        else:
            y_hat = np.where(y_hat > 0, 1, -1)

        # return the predicted value/values
        return y_hat