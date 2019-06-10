# AFL Deep learning models

Aim is to discover the application of neural networks in AFL games and explore sets of features that can be used for predicting 
the current game



# Method

Two datasets were created to explore which features are useful for predicting match outcomes 
using two separate neural networks. The first model was tested on a Rating Systems dataset (11 features) 
which is an Elo model with Home Advantage. The Elo dataset is based on an optimisation created in Excel solver using a regression 
towards the mean for each new season to reset the Elo. 

The second model was tested on a Performance based dataset which contains the average performance in a previous match (21 features). 
The neural network was created with Python using Keras . Grid Search cross fold validation used to find optimal hyper-parameters for both models and a Drop Out layer added to eliminate overfitting



