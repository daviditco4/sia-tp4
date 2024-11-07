# Task 1.1
This task contains an implementation of a Kohonen Self-Organizing Map (SOM) in Python, a type of unsupervised learning algorithm primarily used for clustering and visualizing high-dimensional data.

## System requirements
* Python 3.7+


## Configuration
Customize the SOM parameters in prototype.json:
* width and height: Dimensions of the SOM grid.
* radius: The initial radius of the neighborhood function.
* learning_rate: The learning rate of the SOM.
* distance_method: The method used to calculate the distance between neurons.
* iterations: The number of iterations to train the SOM.
* decay_rate: The rate at which the radius decay.
* init_from_data: Whether to initialize weights based on the input data.

## How to use
 Clone or download this repository in the folder you desire
* In a new terminal, navigate to the `task1_kohonen` repository using `cd`
* When you are ready, enter a command as follows:
```sh
python3 kohonen.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.