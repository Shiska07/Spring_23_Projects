This project implements a simple multi-layer neural network from scratch using NumPy. No other non-standard libraries have been used. 
 
- Weights are adjusted using stochastic gradient descent.
- Bias in included in the weights matrix as the first column.
- MSE is used to calculate error.
- Partial derivatives are calculated using the centered difference approximation method.

Returns:
- A list of final weights matices, an array of MSE values after each epoch, predicted values for 'X_test'