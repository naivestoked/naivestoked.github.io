(This optional walkthrough is adapted from the assignment "Logistic Regression with a Neural Network mindset" in the Neural Networks course from DeepLearning.AI Specialization, and uses common machine learning notation. The first part (Logistic Regression) was transcribed from web sources and structured using the assistance of an LLM (Claude Sonnet 4.5); the second part (two layer NN) was created by the instructor, who usedan LLM to repeat the sections and adapt equations from the first part. The entirety of the text was edited and reviewed by the instructor.)

### The Central Insight


Logistic regression can be understood as the simplest possible neural network. It contains all the components of larger, deep neural networks, which replicate the computations shown here to several nodes per layer in one or more layers.


### Setting Up the Problem
Consider a binary classification task, like determining whether an image contains a cat or not. This Machine Learning problem can be solved using a Neural Network of any size as well as a traditional logistic regression. Suppose we want to start with a logistic regression, but scale up to a neural network if a larger model is needed, but without having to formalize two completely different models.
Each training example (i.e. an observation) consists of an input file (represented as a flattened vector of pixel values, e.g. for a 16x16 pixel image we have a single column of 256 pixel values) and a label: 1 for cat, 0 for non-cat.


### Logistic Regression as a One-Neuron Neural Network structure


Any classification method needs to receive the all the input features (pixel values), perform a computation, and output a prediction of a probability (i.e. a value between 0 and 1) that the input belongs to the positive class (cat, in our example).


1. Linear Combination (Computing $z$)


The first step is to compute a weighted sum of the inputs. For a single training example (i.e. a single image) , we calculate:


The result  is a single number that represents the neuron's pre-activation value — 
this is similar to computing the output of a linear model for a single data point , but here the parenthesis index notation is used to avoid confusion with other indices . Also, it is exactly the same quantity computed for any neurons in the first layer of a neural network: each input feature is multiplied by its corresponding weight (i.e. parameter in the parameter vector ), these products are summed together, and the "bias" term is added — that's all Machine Learning jargon for computing a slope using a dot product and adding a scalar intercept.


2. Activation Function (Computing $a$)


The linear combinations can produce any real number, but we need our output to represent a probability. This is where the logistic function comes in:


The sigmoid function smoothly maps any real number to the range . When  is large and positive,  approaches 1, when it's large and negative, it approaches 0; when  is near zero,  approaches near 0.5. This output  is the same as the logistic model prediction  — the model's estimate of the probability that example  belongs to the positive class.


3. Defining the Likelihood/Cost Function


To estimate the weights, we need to compute the likelihood of the parameters in the logistic model given the data. For the Bernoulli likelihood, the expression for several observations is the following

In plain english, it says the probability of the a cat-positive image is given by the probability  and that of a cat-negative image is given by (because the observation  will either be zero or one, either one of the terms will be 1 and not affect the likelihood (try writing it out for a random probability  and label values of zero and one).
The statistical modeling approach would be to compute the likelihood for all observations (i.e. the likelihood of the model parameters given the data), and use an inference algorithm to maximize the likelihood (or log-likelihood) or sample a distribution of parameter values. The equivalent Machine Learning approach is to call it a loss function, as

This is completely equivalent to taking the log of the likelihood (by logarithm properties exponents end up multiplying each term, and products and divisions become addition or subtraction). The common practice is to take the average of all data points, but that is not necessary, we can use the sum as well. In Machine Learning, this optimization is commonly done by gradient descent algorithms.



# An actual deep Neural Network using the same mindset

## Network Architecture

Now that we formulated the logistic regression in this particular, step-by-step description, we're going to use the same "mindset" to formalize a deep neural network (albeit small) with two layers and three neurons per layer. 

In Deep Learning terms we would say this _architecture_ consists of:

- **Input layer**: receives the feature vector $x$ with $n$ features
- **Hidden layer 1**: contains 3 neurons with ReLU activation
- **Hidden layer 2**: contains 3 neurons with ReLU activation  
- **Output layer**: contains 1 neuron with sigmoid activation for binary classification

This gives us a 3-layer network (we typically count only the layers with parameters: the two hidden layers plus the output layer).

## Notation

For a network with $L = 3$ layers:
- $n^{[0]} = n$: number of input features (i.e. pixel values)
- $n^{[1]} = 3$: number of neurons in first hidden layer
- $n^{[2]} = 3$: number of neurons in second hidden layer
- $n^{[3]} = 1$: number of neurons in output layer (binary classification)
- $W^{[l]}$: weight matrix for layer $l$
- $b^{[l]}$: bias vector for layer $l$
- $Z^{[l]}$: pre-activation (linear combination) for layer $l$
- $A^{[l]}$: activation (post-activation) for layer $l$
- $A^{[0]} = X$: the input

## Forward Propagation

Forward propagation is neural network jargon for computing the value of an explicit function, same as a linear model or any model that is given by a function. It gets its name from the fact that the linear combinations and activations of neurons in the neural network layers need to be computed sequentially, so they would be "propagating" like electrical impulses in the nervous system.
That's also useful for a description consistent with the steps in the NN mindset we used before, so we're going to look at individual neurons, then show how they com together


**Layer 1 (First Hidden Layer) - Individual Neurons**

The first hidden layer contains 3 neurons. Each neuron operates exactly like the logistic regression neuron, computing a weighted sum followed by an activation function.

*Neuron 1:*
For a single training example $x^{(i)}$, the first neuron computes:

$$z_1^{[1](i)} = w_1^{[1]T} x^{(i)} + b_1^{[1]}$$

where $w_1^{[1]}$ is the weight vector for neuron 1 (dimension $n \times 1$) and $b_1^{[1]}$ is its bias (a scalar). The subscript and square-bracket sueprscript are new here, they represent the neuron number and layer, respectively, and could be ommitted from the logistic regression because there was only one node, so we didn't neet to keep track of which activations we were computing. Now we do.

The activation using the logistic function is:

$$a_1^{[1](i)} = \sigma(z_1^{[1](i)})

*Neuron 2:*

Similarly, the second and third neurons compute the following activation functions:

$$z_2^{[1](i)} = w_2^{[1]T} x^{(i)} + b_2^{[1]}$$

$$a_2^{[1](i)} = \sigma(z_2^{[1](i)})$$

*Neuron 3:*

$$z_3^{[1](i)} = w_3^{[1]T} x^{(i)} + b_3^{[1]}$$

$$a_3^{[1](i)} = \sigma(z_3^{[1](i)})$$

Each neuron performs the same two-step computation as in logistic regression: $z = w^T \cdot (\text{input}) + b$, then apply activation. The difference these neurons feed into the next layer rather than producing final predictions.

**Layer 2 (Second Hidden Layer) - Individual Neurons**

The second hidden layer also has 3 neurons. Each neuron takes as input the activations from layer 1, that is $a^{[1](i)} = [a_1^{[1](i)}, a_2^{[1](i)}, a_3^{[1](i)}]^T$ (a 3-dimensional vector for each example) replaces $x$ as their input -- so the dimension of the input, and as a consequence the number of parameters in subsequent layers, can vary. Other than that we have the same kind of calculations:

*Neuron 1:*

The first neuron in layer 2 computes:

$$z_1^{[2](i)} = w_1^{[2]T} a^{[1](i)} + b_1^{[2]}$$

where $w_1^{[2]}$ is a 3-dimensional weight vector (since it receives 3 inputs from layer 1) and $b_1^{[2]}$ is a scalar bias.

The activation is:

$$a_1^{[2](i)} = \sigma(z_1^{[2](i)})$$

*Neuron 2:*

$$z_2^{[2](i)} = w_2^{[2]T} a^{[1](i)} + b_2^{[2]}$$

$$a_2^{[2](i)} = \sigma(z_2^{[2](i)})$$

*Neuron 3:*

$$z_3^{[2](i)} = w_3^{[2]T} a^{[1](i)} + b_3^{[2]}$$

$$a_3^{[2](i)} = \sigma(z_3^{[2](i)})$$

Again, each neuron performs the exact same computation: weighted sum plus bias, then activation. The input has changed (now it's $a^{[1]}$ instead of $x$), but the operation is identical.

**Layer 3 (Output Layer) - Single Neuron**

Finally, we have a single output neuron that performs binary classification, exactly like logistic regression. The difference here is the input to this binary neuron is no longer the original features-times-weights through an activation sigmoid, but rather they had to be propagated through the two hidden layers (it may be evident that we can make the size and number of layers whatever we would like, and the computations would be repeated in a similar way).

*The Output Neuron:*

This neuron takes the 3 activations from layer 2 as input:

$$z^{[3](i)} = w^{[3]T} a^{[2](i)} + b^{[3]}$$

where $w^{[3]}$ is a 3-dimensional weight vector and $b^{[3]}$ is a scalar.

We apply the sigmoid activation to get a probability (again, boringly, same as in the logistic regression):

$$a^{[3](i)} = \sigma(z^{[3](i)}) = \frac{1}{1 + e^{-z^{[3](i)}}}$$

This binary classification node computes $z = w^T \cdot (\text{input}) + b$, then applies sigmoid. The only difference is that its input is $a^{[2]}$ (the learned features from the hidden layers) instead of the raw features $x$.

## Matrix Notation
Many of these computations can be optimized or computed in parallel, this explainer won't go into that aspect of calculations, but from a point of view of notation compactness many of these calculations can be expressed as matrix calculations instead of node-by-node computations:

**Layer 1**

We can stack the 3 neurons efficiently using matrices:

$$W^{[1]} = \begin{bmatrix} - & w_1^{[1]T} & - \\ - & w_2^{[1]T} & - \\ - & w_3^{[1]T} & - \end{bmatrix}, \quad b^{[1]} = \begin{bmatrix} b_1^{[1]} \\ b_2^{[1]} \\ b_3^{[1]} \end{bmatrix}$$

For all $m$ training examples simultaneously:

$$Z^{[1]} = W^{[1]} X + b^{[1]}$$

where $W^{[1]}$ is $(3 \times n)$, $X$ is $(n \times m)$, and $b^{[1]}$ is $(3 \times 1)$.

Applying ReLU element-wise:

$$A^{[1]} = g^{[1]}(Z^{[1]}) = \max(0, Z^{[1]})$$

This produces a $(3 \times m)$ matrix where row $k$ contains activations for neuron $k$ across all examples.

**Layer 2 - Matrix Notation**

Stacking into matrix form:

$$W^{[2]} = \begin{bmatrix} - & w_1^{[2]T} & - \\ - & w_2^{[2]T} & - \\ - & w_3^{[2]T} & - \end{bmatrix}, \quad b^{[2]} = \begin{bmatrix} b_1^{[2]} \\ b_2^{[2]} \\ b_3^{[2]} \end{bmatrix}$$

For all examples:

$$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$$

where $W^{[2]}$ is $(3 \times 3)$ and $b^{[2]}$ is $(3 \times 1)$.

Applying ReLU:

$$A^{[2]} = g^{[2]}(Z^{[2]}) = \max(0, Z^{[2]})$$

**Layer 3 - Matrix Notation**

For all examples:

$$Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]}$$

where $W^{[3]}$ is $(1 \times 3)$ and $b^{[3]}$ is a scalar.

Applying sigmoid:

$$A^{[3]} = g^{[3]}(Z^{[3]}) = \sigma(Z^{[3]}) = \frac{1}{1 + e^{-Z^{[3]}}}$$

This produces predictions $\hat{y}^{(i)} = a^{[3](i)}$ for each example.

## The Loss Function: Binary Cross-Entropy

Since we have a single output neuron performing binary classification, we use the same loss function as logistic regression, derived from the Bernoulli likelihood.

**Bernoulli Likelihood**

For a binary outcome $y^{(i)} \in \{0, 1\}$, the Bernoulli probability mass function is:

$$P(Y = y^{(i)}) = \begin{cases} 
p & \text{if } y^{(i)} = 1 \\
1-p & \text{if } y^{(i)} = 0
\end{cases}$$

In our network, the success probability is our output neuron's prediction $a^{[3](i)} = \sigma(w^{[3]T} a^{[2](i)} + b^{[3]})$. The likelihood of observing label $y^{(i)}$ is:

$$P(y^{(i)} | x^{(i)}, \theta) = \begin{cases} 
a^{[3](i)} & \text{if } y^{(i)} = 1 \\
1 - a^{[3](i)} & \text{if } y^{(i)} = 0
\end{cases}$$

where $\theta$ represents all parameters $(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}, W^{[3]}, b^{[3]})$.

**Log-Likelihood**

Assuming independence, the likelihood over all $m$ examples is:

$$L(\theta) = \prod_{i=1}^{m} P(y^{(i)} | x^{(i)}, \theta)$$

Taking the logarithm:

$$\log L(\theta) = \sum_{i=1}^{m} \log P(y^{(i)} | x^{(i)}, \theta)$$

For each term, when $y^{(i)} = 1$, we have $\log(a^{[3](i)})$; when $y^{(i)} = 0$, we have $\log(1 - a^{[3](i)})$. This can be written compactly as:

$$\log L(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(a^{[3](i)}) + (1 - y^{(i)}) \log(1 - a^{[3](i)}) \right]$$

**Cost Function**

The cost function is the negative log-likelihood, averaged over examples:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(a^{[3](i)}) + (1 - y^{(i)}) \log(1 - a^{[3](i)}) \right]$$

For a single example:

$$\mathcal{L}(a^{[3](i)}, y^{(i)}) = -\left[ y^{(i)} \log(a^{[3](i)}) + (1 - y^{(i)}) \log(1 - a^{[3](i)}) \right]$$

This is identical to the logistic regression loss function—minimizing this cost is equivalent to maximizing the likelihood of our training data under the Bernoulli model.

## Backpropagation

Backpropagation computes gradients by applying the chain rule backward through the network.

**Output Layer Gradients (Layer 3)**

Starting from the output, the gradient with respect to the pre-activation is:

$$dZ^{[3]} = A^{[3]} - Y$$

where $Y$ is the $(1 \times m)$ vector of true labels. This simple form comes from the combination of sigmoid and binary cross-entropy.

The parameter gradients are:

$$dW^{[3]} = \frac{1}{m} dZ^{[3]} (A^{[2]})^T$$

$$db^{[3]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[3](i)}$$

**Second Hidden Layer Gradients (Layer 2)**

Propagate backward to layer 2:

$$dA^{[2]} = (W^{[3]})^T dZ^{[3]}$$

Apply ReLU derivative:

$$dZ^{[2]} = dA^{[2]} \odot g'^{[2]}(Z^{[2]})$$

where:

$$g'^{[2]}(Z^{[2]}) = \begin{cases} 
1 & \text{if } Z^{[2]} > 0 \\
0 & \text{if } Z^{[2]} \leq 0
\end{cases}$$

Parameter gradients:

$$dW^{[2]} = \frac{1}{m} dZ^{[2]} (A^{[1]})^T$$

$$db^{[2]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[2](i)}$$

**First Hidden Layer Gradients (Layer 1)**

Propagate to layer 1:

$$dA^{[1]} = (W^{[2]})^T dZ^{[2]}$$

Apply ReLU derivative:

$$dZ^{[1]} = dA^{[1]} \odot g'^{[1]}(Z^{[1]})$$

Parameter gradients:

$$dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T$$

$$db^{[1]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[1](i)}$$

## Parameter Updates

Using gradient descent, we update all parameters:

$$W^{[l]} := W^{[l]} - \alpha \, dW^{[l]} \quad \text{for } l = 1, 2, 3$$

$$b^{[l]} := b^{[l]} - \alpha \, db^{[l]} \quad \text{for } l = 1, 2, 3$$

where $\alpha$ is the learning rate.

## Summary: From Logistic Regression to Deep Networks

This network demonstrates the fundamental insight: **every neuron in a neural network is performing the exact same computation as the logistic regression neuron** ($z = w^T \cdot \text{input} + b$, then activation).

The progression from logistic regression to this deeper network:

1. **Logistic regression**: 1 neuron, raw features → sigmoid → prediction
2. **This network**: 
   - Layer 1: 3 neurons transform raw features using ReLU
   - Layer 2: 3 neurons combine layer 1 features using ReLU  
   - Layer 3: 1 neuron (exactly like logistic regression) produces final prediction

Each hidden layer learns increasingly abstract representations. Layer 1 might detect edges or simple patterns, layer 2 combines these into more complex features, and the output neuron uses these learned features for classification. But at every step, each individual neuron is just computing a weighted sum plus bias, then applying an activation—the same operation we started with in logistic regression.