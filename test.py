from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from tensorflow.keras.datasets import fashion_mnist

print("GPU available:", tf.config.list_physical_devices('GPU'))
print("Is GPU available?", tf.test.is_gpu_available())

