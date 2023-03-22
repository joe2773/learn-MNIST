import numpy as np
import pickle
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')

# save the data to a local file
with open('mnist.pkl', 'wb') as f:
    pickle.dump(mnist, f)
