# Convolutional Model

This is a python script that train, test and generate a convolutional model based on Mnist datasets.


# Getting started

## Step 0 (Requirements)

* Python version 3.6.x [https://www.python.org/downloads/](https://www.python.org/downloads/)
* Check python installation `python -V` or `python3 -V` _(It depends on your OS)_
* Install Tensorflow by using Python Package Index (Pip) `pip3 install tensorflow`

## Step 1

You can configure the script parameters by changing those variables on the `model.py`
```python 
# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units
```

Then, launch the python script `python model.py` or `python3 model.py` _(It depends on your OS)_ to start training the model and test it. 

**OUTPUTS**

* You should get as output the Iteration number, the cost and when the script finishes it gives you the accuracy on a test datasets.

* In addition to that you should get two new folders created `logs` and `saved-models/convolutional/`.

	* The `logs` contains the summary that you can view on tensorboard by runing this command `tensorboard --logdir=/path/to/this/folder/logs` 

	* The `saved-models/convolutional/` contains the saved model that we can use on java application. The path you should give to **java application** to replace the `PATH_TO_CONVOLUTIONAL_MODEL` variable is `/path/to/this/folder/saved-models/convolutional/`

