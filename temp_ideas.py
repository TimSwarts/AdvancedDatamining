# ipv de for loops in de dense layer, is deze list comprehension mogelijk:
# Calculate pre activation values: linear combination for all inputs for every neuron for all instances
# aa = [[self.bias[o] + sum(self.weights[o][i] * x[i] for i in range(self.inputs))
#        for o in range(self.outputs)]
#       for x in xs]


# short plotting code for insight into these Linear Unit Functions:
# import matplotlib.pyplot as plt
# import numpy as np

# Define the functions
# def relu(x):
#     return np.maximum(0, x)
#
# def elu(x, alpha=1):
#     return np.where(x > 0, x, alpha * (np.exp(x) - 1))
#
# def selu(x, alpha=1.67326, scale=1.0507):
#     return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def silu(x):
#     return x * sigmoid(x)
#
# # Generate x values
# x = np.linspace(-5, 5, 100)
#
# # Plot the functions
# plt.plot(x, relu(x), label='ReLU')
# plt.plot(x, elu(x), label='ELU')
# plt.plot(x, selu(x), label='SELU')
# plt.plot(x, silu(x), label='SiLU')
#
# # Add a legend
# plt.legend()
#
# # Show the plot
# plt.show()


# softmax
# from math import e
# h = [-3, 4.68, 0.34, 1.0]
#
# def softmax(h):
#     denominator = sum(e**hi for hi in h)
#     y = [e**ho / denominator for ho in h]
#     return y
#
# y = softmax(h)
# print(y, sum(y))

#
# def signum1(a):
#     if a > 0:
#         return 1
#     elif a < 0:
#         return -1
#     else:
#         return 0
#
#
# def signum2(a):
#     if a > 0:
#         return 1
#     if a < 0:
#         return -1
#     return 0
#
#
# def signum3(a):
#     return -1 if a < 0 else 1 if a > 0 else 0
#
#
# values = [-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4]
#
# def call_signum1(values):
#     for a in values:
#         print(signum1(a))
#
# def call_signum2(values):
#     for a in values:
#         print(signum2(a))
#
# call_signum1(values)
# call_signum2(values)
