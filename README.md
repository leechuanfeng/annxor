# annxor
Familiarizing with the principle of feed-forward artificial neural network and gradient descent method for learning in a multi-layered network, as well to implement the Back Propagation algorithm to learn parameters of FF ANN for solving the XOR problem. 

To try out this project, run the following lines in cmd in sequence:

Instructions:

```sh
$ python
$ import os
$ os.chdir("C:\Users\username\Desktop\py-annxor")
$ import annxor
$ annxor.run()
```

### Abstract
Feed-forward Artificial Neural Network is based on a collection of connected units or nodes called artificial neurons which is inspired by the biological neural networks, it learns to perform tasks by considering examples without any prior knowledge, by automatically generating identifying characteristics from the learning material that they process

### Dataset Information

1. XOR Truth Table
- 2 features  [0 0 1 1], [0 1 0 1]
- 1 output    [0 1 1 0]

### ANN for XOR
![fig1](https://github.com/leechuanfeng/annxor/blob/master/images/fig1.png "fig1")

### Feed Forward Equation
To begin, we have to initialize the parameters, the weights and the bias that are required for the network. In order for this to work, we have to randomly generate a total of 9 parameters, 6 weight parameter and 3 bias. Along with these, we have to provide 2 input.

![fig1.2](https://github.com/leechuanfeng/annxor/blob/master/images/fig1.2.png "fig1.2")
Let’s look into it in more details below.

![fig2](https://github.com/leechuanfeng/annxor/blob/master/images/fig2.png "fig2")

The diagram above demonstrate a simple example showing the stages of processing the input X1=0, X2=1. Upon understanding the process, we can make use of matrix multiplication to aid us in quickly computing the result using all the inputs.
We can observe that the matrix Wh = [[b1 w1,1 w1,2], [b2 w2,1 w2,2]] consist of the initial bias value and weights needed to compute the value for our hidden layer. As for matrix Wo = [b3 w3,1 w3,2], it consists of the bias and weights for the computation in our output layer.
As we can see above, we used tanh as our activation function to compute the value of the hidden layer from our input (X1, X2) and sigmoid for the output layer. Applying so will compute the result for feed forward.

### Back Propagation

![fig2.2](https://github.com/leechuanfeng/annxor/blob/master/images/fig2.2.png "fig2.2")

During back propagation we use our current value compared to our actual result to recalculate the new bias and weights that will help us get the result closer to the actual values.

### Experimental Result

![fig3](https://github.com/leechuanfeng/annxor/blob/master/images/fig3.png "fig3")
