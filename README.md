# CNTK UWP Example

A simple example demonstrating the use of CNTK in a UWP app. The solution provides an end-to-end example from building the model in Python to using the model in a UWP app via a C++/CX wrapper class.

This solution includes the following projects:

- CNTKFitExample - A Python script that uses CNTK to build, fit, and export a simple linear model.
  - The script creates a list of random features and labels that are dependent on the features i.e. label = 1 if w1 * feature1 + w2 * feature2 >= t else 0.
  - The function create_model create a simple model of the form label = sigmoid(feature * W + b).
  - The trainer uses the stochastic gradient descent method to minimise the square of the error i.e minimise (prediction - label)^2.
  - Finally, the model is exported for the UWP app to pick and import.
- CNTKLib - A C++/CX wrapper to provide access to the model from C#.
  - The library provides the class CNTKModel which has a constructor that takes in the file location of model and a method GetPrediction which returns the models prediction for the provided input.
- CNTKUWPApp - A minimal UWP app that demonstrates using the C++/CX wrapper from C#.

