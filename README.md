# audio-mps
Audio Synthesis with Matrix Product States

## Software

* TensorFlow
* Python 3.6.6

## Repository contents

* model.py

It defines our RNN model and contains other methods, to sample from the trained model for example.

* training_estimators.py

It loads the training data, and runs the training. A custom estimator is used.

* reader.py

It picks out a subset of the data, that contains a desired feature: pitch, instrument ...

* make-small-dataset.py

It imports reader.py to make the dataset.

## Todo

1. ~~What about initialization? Glorot variance depends on size. Change to uniform~~
1. Sampling
1. Interaction picture 
1. Prepare spectrograph code to look at effect of upsampling (reducing delta_t)
1. Monitor change of state upon update to see if it is small.
1. Exploring latent space
