#!usr/bin/env python3
#-*- coding: utf-8 -*-
"""
'model_plots.py'

Plot learning curves, given output data from scikit-learn's 'learning_curve' function

2018 Steve Neale <steveneale3000@gmail.com>

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(train_sizes, train_mean, train_std, test_mean, test_std):
    """ Plot a learning curve """

    # Plot training accuracy means for a given series of training sizes
    plt.plot(train_sizes, train_mean, color="blue", marker="o", markersize=5, label="Training accuracy")
    # Add a coloured fill showing the standard deviation of the training accuracy for a given series of training sizes
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color="blue")
    # Plot test accuracy means for a given series of training sizes
    plt.plot(train_sizes, test_mean, color="green", linestyle="--", marker="s", markersize=5, label="Test accuracy")
    # Add a coloured fill showing the standard deviation of the test accuracy for a given series of training sizes
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color="green")
    # Add gridlines to the plot
    plt.grid()
    # Add captions to the X and Y axes of the plot
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    # Provide a location for the plot's legend/key
    plt.legend(loc="lower right")
    # Set upper and lower limits on the y axis
    plt.ylim([0.8, 1.0])
    # Show the plot
    plt.show()