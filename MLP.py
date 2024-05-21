import numpy as np
import copy

# Class which handles all of the perceptron magic
class MLP():
    """ # Multilayer Perceptron
    A class containing code for a multiple layer perceptron.
    """
    
    def __init__(self, Ws, eta=0.001):
        self.Ws = Ws
        self.no_layers = len(Ws)
        self.eta = eta

    # This is the actual function that trains the model as per instructions
    def train_model(self, dataset, val_dataset, num_epochs=1000):
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
        X = dataset[["x1", "x2"]].to_numpy()   # features (called patterns in assignment)
        X = np.hstack((X, np.ones((X.shape[0], 1)))) # add bias term
        
        X_val = val_dataset[["x1", "x2"]].to_numpy()   # features (called patterns in assignment)
        X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1)))) # add bias term
        
        y = dataset["y"].to_numpy().reshape(-1,1)            # labels (called targets in assignment)
        
        y_val = val_dataset["y"].to_numpy().reshape(-1,1)            # labels (called targets in assignment)
        
        acc_history = []
        MSE_history = []
        val_MSE_history = []
        
        # List for saving weights after each epoch
        Ws_history = []
        
        # Actual training
        for epoch in range(num_epochs):
            self.Ws = self._backprop(X, y, self.Ws)
            weight_copy = copy.deepcopy(self.Ws)
            Ws_history.append(weight_copy)

            
            # calculate model training mean square error
            y_hat = self._forward(X, self.Ws)[-1]
            
            squared_errors_list = (y - y_hat) ** 2    # assuming y, predictions are numpy arrays
            MSE = squared_errors_list.mean()
            MSE_history.append(MSE)
            
            # calculate validation mean square error
            y_hat_val = self._forward(X_val, self.Ws)[-1]
            
            squared_errors_list_val = (y_val - y_hat_val) ** 2    # assuming y, predictions are numpy arrays
            val_MSE = squared_errors_list_val.mean()
            val_MSE_history.append(val_MSE)                        

            # generate model predictions
            predictions = self._step(y_hat)

            # calculate model accuracy
            correct_predictions = np.sum(y == predictions)
            accuracy = correct_predictions / len(y)
            acc_history.append(accuracy)

        return Ws_history, acc_history, MSE_history, val_MSE_history
    

    def predict(self, X:np.ndarray, Weights=None):
        """ # predict
        Predicts the labels for the given input features.

        Args:
            X (np.ndarray): Input features.
            Weights: Optional weights to use for prediction

        Return:
            y_hat (np.ndarray): Predicted labels.
        """
        # Add bias term to X
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        
        if Weights is not None:
            z = self._forward(X, Weights)
        else:
            z = self._forward(X, self.Ws)
        y_hat = z[-1]
        return self._step(y_hat)
        

    # Private functions for training the perceptron
    def _backprop(self, X, y, Ws):
        """ # _backprop
        Backpropagation algorithm for updating the weights of the multilayer perceptron.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target labels.
            Ws (list): List of weight matrices for each layer in the network.

        Return:
            Ws (list): Updated list of weight matrices for each layer in the network.
        """

        # Forward pass
        z = self._forward(X, Ws)

        # Backward pass
        deltas = self._backward(X, y, z, Ws)
        
        # Update weights
        for i in range(len(Ws)):
            if i == 0:
                Ws[i] -= self.eta * (deltas[i].T @ X)[:-1,:].T
            else:
                Ws[i] -= self.eta * (deltas[i].T @ z[i-1]).T

        return Ws
        
    def _forward(self, X, Ws):
        """ # _forward
        Forward pass of the multilayer perceptron.

        Args:
            X (np.ndarray): Input features.
            Ws (list): List of weight matrices for each layer in the network.

        Return:
            z (list): List of activation values for each layer in the network.
        """
        z = []
        for i, W in enumerate(Ws):
            if i == 0:
                # Input layer
                # Add bias term to X
                z.append(self._sigmoid(X @ W))            
            else:
                # Add bias term to z[i-1]
                z[i-1] = np.hstack((z[i-1], np.ones((z[i-1].shape[0], 1))))
                z.append(self._sigmoid(z[i-1] @ W))
                
        return z
    
    def _backward(self, X, y, z, Ws):
        deltas = [None] * len(Ws)
        for i in range(len(Ws)-1, -1, -1):
            if i == len(Ws)-1:
                # Output layer
                deltas[i] = ((z[i] - y) * self._sigmoid_derivative(z[i]))
            else:
                deltas[i] = ((Ws[i+1] @ deltas[i+1].T).T * self._sigmoid_derivative(z[i]))
                            
        return deltas
    
    # Transfer function
    def _sigmoid(self, x):
        return 2 / (1 + np.exp(-x)) - 1
    

    # Transfer function derivative
    def _sigmoid_derivative(self, sigmoid_x):
        return np.multiply((1 + sigmoid_x), (1 - sigmoid_x))/2
    
    def _step(self, y_hat):
        # Predict for single number
        if isinstance(y_hat, (int, float)):
            y_hat = 1 if y_hat > 0 else -1

        # Predict for array
        else:
            y_hat = np.where(y_hat > 0, 1, -1)

        # return the predicted value/values
        return y_hat
    
