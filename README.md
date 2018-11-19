# audio-mps
Audio Synthesis with Matrix Product States

## Software

* TensorFlow
* Python 3.6.6

## Repository contents

* model.py

It defines our RNN model and contains other methods, to sample from the trained model for example.

* training.py

It loads the training data, and runs the training.

* reader.py

It picks out a subset of the data, that contains a desired feature: pitch, instrument ...

* make-small-dataset.py

It imports reader.py to make the dataset.
