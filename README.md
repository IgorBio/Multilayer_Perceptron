# Multilayer Perceptron v1.0
The "Multilayer Perceptron" is a modern app written in C++. The application represents a simple artificial neural network in the form of a perceptron, which can be trained on an open dataset and perform recognition of 26 handwritten letters of the Latin alphabet.

#### Program launch options

```
make run
```

## Features
- GUI implementation, based on QT6

  ![MLP GUI Screenshot](./src/docs/images/GUI.png)

- Load train and test datasets from a csv file.
- Choose the network topology with 2-5 hidden layers.
- Training with using the backpropagation method and sigmoid activation.
- Matrix form: all layers are represented as weight matrices.
- Graph form: each neuron is represented as some node object connected to other nodes by refs.
- Perform experiments on a selected portion of the test sample, defined by a floating-point number ranging from 0 to 1.
- Load BMP images (image size can be up to 512x512) with Latin letters and classify them.
- Draw two-color square images by hand and classify them.
- Real-time training process for a user-defined number of epochs with displaying the error values for each training epoch.
- Run the training process using cross-validation for a given number of groups k.
- Save to a file and load weights of perceptron from a file.

  ![MLP Recognition Screecast](./src/docs/images/Recognition.gif)

## License
Copyright (c). All rights reserved.