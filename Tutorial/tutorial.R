#-------------------------------------------------------------------------------
#===============================================================================
#
# DIGIT RECOGNITION: BUILDING A NEURAL NETWORK FROM SCRATCH
#
#===============================================================================
#-------------------------------------------------------------------------------


#===============================================================================
# Introduction
#===============================================================================

# This is a step-by-step tutorial for creating a neural network, from scratch,
# to recognise handwritten digits from the MNIST dataset - the basis of Kaggle's
# Digit Recognizer competition.

# Owen Jones | Bath Machine Learning Meetup



#===============================================================================
# Section 0: Getting the data, installing packages ----
#===============================================================================

# Kaggle competition:
"https://www.kaggle.com/c/digit-recognizer"

# Download CSVs "train.csv" and "test.csv" from here:
"https://www.kaggle.com/c/digit-recognizer/data"

# Install packages we'll need:
install.packages("ggplot2", "reshape2")



#===============================================================================
# Section 1: Loading and organising data ----
#===============================================================================

# Read training data from CSV into R
train <- read.csv("C:/Users/Desktop/neuralnet/data/train.csv")

# Let's have a look
head(train)

# Woah OK, lots of info!
# First column is the label (0 to 9) of what the image is. Next 784 columns are
# pixel intensity values for the "unrolled" 28x28 image.

# All info is numeric so let's turn the data frame into a matrix
train <- as.matrix(train)

# If we look at the first column (the labels) the set already seems to be 
# shuffled (i.e. not all the zero images, followed by all the ones, followed by
# all the twos etc.) but to be on the safe side we'll shuffle it anyway!

train <- train[sample(1:42000), ]

# Let's separate the labels from the image data
labels <- train[, 1]
train <- train[, -1]

# We'll make a couple of adjustments to make later calculations work better!

# First we'll re-label "0" as "10":
labels[labels == 0] <- 10

# And we'll rescale pixel intensity from 0 to 1, instead of 0 to 255:
range(train)
train <- train / max(train)
range(train)

# Now we'll split the examples in a 60:20:20 ratio: a training set, a
# cross-validation set (for evaluating performance and adjusting
# hyper-parameters) and a test set (for evaluating overall performance)

ytrain <- labels[1:25200]
Xtrain <- train[1:25200, ]

yval <- labels[25201:33600]
Xval <- train[25201:33600, ]

ytest <- labels[33601:42000]
Xtest <- train[33601:42000, ]



#===============================================================================
# Section 2: Visualising data ----
#===============================================================================

# We'll make a function to turn the numbers back into an image we can 
# understand!

# We need the "melt" funciton from reshape2, and we'll use ggplot2 to make a
# plot
library(reshape2)
library(ggplot2)

# Quick demo of melt
A <- matrix(1:9, 3)
A
melt(A)

# Let's make our function!

visualise <- function(imgvec) {
    
    n <- length(imgvec)
    
    # Reshape into a matrix...
    img <- matrix(imgvec, sqrt(n))
    # ... then melt it!
    imgmelt <- melt(img)
    
    ggplot(imgmelt, aes(x = Var1, y = -Var2, fill = value)) +
        geom_raster() +
        scale_fill_gradient(low = "white", high = "black")
}

# Now let's have a proper look at some of our data
visualise(Xtrain[1, ])
visualise(Xtrain[50, ])



#===============================================================================
# Section 3: Network structure and parameters ----
#===============================================================================

# We're going to build a fully-connected, single-hidden-layer network. There are
# three layers in total:
# * The "input" layer is our 784 pixels
# * The "hidden" layer is how many "features" the net will learn to recognise
# * The "output" layer is the classification - i.e. what number the net decides
#   the image is

# How does the net learn?
# By adjusting how much each pixel "counts" towards the classification. To start
# with, we just guess!

# We'll set up some matrices: one for each "transition" between layers

# Choose a small value to define a range for initial values (this one seems to
# work well!)
# sqrt(6) / (sqrt(input_layer_size) + sqrt(output_layer_size))
epsilon <- sqrt(6) / (sqrt(784) + sqrt(10))

# We'll set a "hyper-parameter" for the number of neurons in the hidden layer,
# so we can adjust it later if we want to

hidden_size <- 25

# Number of parameters in each matrix
n1 <- 785 * hidden_size
n2 <- (hidden_size + 1) * 10

# Generate enough random values to fill both
init_params <- runif((n1+n2), min = -epsilon, max = epsilon)

# It will save time if we write a small function to reshape these parameters
# into matrices, rather than doing it by hand each time

make_thetas <- function(paramvec) {
    
    thetas <- list(type = "list", length = 2)
    
    thetas[[1]] <- matrix(paramvec[1:n1], hidden_size, 785)
    
    thetas[[2]] <- matrix(paramvec[(n1+1):(n1+n2)], 10, (hidden_size + 1))
    
    thetas
}

# So we get an output like this:
thetas <- make_thetas(init_params)
thetas[[2]]



#===============================================================================
# Section 4: Forward propagation and making predictions ----
#===============================================================================

# Question: OK, so we can do some sums - but how do we interpret the results as
# a meaningful output?
# Answer: an "activation" function!
# So the result of the sum either causes the neuron to fire, or doesn't. Or,
# causes it to fire a little bit... so that we know how we can make an
# improvement if we adjust a parameter. So we'll use the "sigmoid" function,
# which is an improvement over an "on-off" step function ("perceptron").
# (Also, unlike a step function the sigmoid function can be differentiated -
# this will be important later!)

sigmoid <- function(z) {
    
    1 / (1 + exp(-z))
}

# What does it look like?
curve(sigmoid(x), from = -10, to = 10, col = "red")

# So large positive sums get mapped to 1, large negative to 0, and small sums
# get mapped more or less linearly in between


# So, let's build a net! We'll make some predictions, and create a way of seeing
# how wrong we are overall.


# First, let's write a function to take our input, run it through the net
# ("forward propagation"), and predict the labels of each input

predict <- function(params, X) {
    
    # Let's make our parameters back into matrices
    thetas <- make_thetas(params)
    Theta1 <- thetas[[1]]
    Theta2 <- thetas[[2]]
    
    # We'll assign a variable equal to the number of examples we're using
    m <- dim(X)[1]
    
    
    # Forward propagation!
    
    # Add a bias unit to each example
    A1 <- cbind(rep(1, m), X)
    
    # Calculate raw output of hidden layer
    Z2 <- A1 %*% t(Theta1)
    
    # Apply the activation function, and add bias unit to each example
    A2 <- cbind(rep(1, m), sigmoid(Z2))
    
    # Calculate raw output of output layer
    Z3 <- A2 %*% t(Theta2)
    
    # Apply activation function: A3 is the overall output of the net
    A3 <- sigmoid(Z3)
    
    
    # Set the column with largest value to be the prediction
    max.col(A3)
}


# So we can make some predictions with our initial parameters! But we haven't
# trained the net yet, so they'll just be random guesses. We'll probably get
# ~10% accuracy.
# IMPORTANT NOTE: never evaluate performance using training data, because if the
# model is overfitting then you'll get an overly-optimistic estimate of how well
# your model is working!

preds <- predict(init_params, Xval)

sum(preds == yval) / length(yval)

# Which is the same as
mean(preds == yval)

# We can make this prettier:
sprintf("Accuracy: %.1f%%", sum(preds == yval) / length(yval) * 100)



#===============================================================================
# Section 5: Cost function ----
#===============================================================================

# OK, so, how do we measure how wrong each prediction was, and how wrong we were
# overall? We quantify the error with a "cost"...

compute_cost <- function(params, X, y, lambda) {
    
    thetas <- make_thetas(params)
    Theta1 <- thetas[[1]]
    Theta2 <- thetas[[2]]
    m <- dim(X)[1]
    
    # Condensed version of the code for forward propagation
    A2 <- sigmoid(cbind(rep(1, m), X) %*% t(Theta1))
    A3 <- sigmoid(cbind(rep(1, m), A2) %*% t(Theta2))
    
    # But how far off are we?
    # If our predictions were 100% perfect, we would end up with this:
    Actual <- diag(10)[y, ]
    
    # Compute the average "difference" between the prediction (the row of A3)
    # and the expected value (the row of Actual). We use log so that as we are
    # further away from correct, the cost becomes exponentially higher.
    J <- sum(-log(A3) * Actual + -log(1 - A3) * (1 - Actual)) / m
    
    # Add regularisation term: we want weights to be small (to prevent
    # overfitting), we add an averaged "squared weight" to discourage large
    # positive/negative weights. The effect of this term is scaled by lambda -
    # higher lambda penalises large weights more heavily.
    J <- J + lambda * (sum(Theta1[, -1] ^ 2) + sum(Theta2[, -1] ^ 2)) / (2*m)
    
    J
}

# So we can have a look at the cost of our initial (bad) predictions. We'll just
# use lambda = 1 for now.
compute_cost(init_params, Xtrain, ytrain, 1)

# So we want to reduce this cost (and therefore improve our predictions)...



#===============================================================================
# Section 6: Backpropagation ----
#===============================================================================

# Backpropagation is how we work out how much each individual parameter is
# contributing to our total cost - we compute the partial derivative (the
# "gradient") of the cost function with respect to each parameter in turn

compute_grad <- function(params, X, y, lambda) {
    
    # First we reshape our parameters, as per usual
    thetas <- make_thetas(params)
    Theta1 <- thetas[[1]]
    Theta2 <- thetas[[2]]
    m <- dim(X)[1]
    
    # Now we forward-propagate to get the network's output
    A1 <- cbind(rep(1, m), X)
    Z2 <- A1 %*% t(Theta1)
    A2 <- cbind(rep(1, m), sigmoid(Z2))
    Z3 <- A2 %*% t(Theta2)
    A3 <- sigmoid(Z3)
    
    
    # Now we calculate the "error" of each unit in each layer, starting with the
    # output layer
    Actual <- diag(10)[y, ]
    
    Delta3 <- A3 - Actual
    
    # Now we propagate this error backwards to the next layer (backpropagation!)
    
    Delta2 <- (Delta3 %*% Theta2[, -1]) * (sigmoid(Z2) * (1 - sigmoid(Z2)))
    
    # The first layer (the input) doesn't have any error, obviously
    
    # So we can compute the partial derivatives of the unregularised cost
    # function:
    Theta1_grad <- (t(Delta2) %*% A1) / m
    Theta2_grad <- (t(Delta3) %*% A2) / m
    
    # And we compute the partial derivative of the regularisation term (note
    # that the bias terms aren't regularised by convention)
    Reg1 <- lambda/m * Theta1
    Reg1[, 1] <- 0
    Reg2 <- lambda/m * Theta2
    Reg2[, 1] <- 0
    
    # So the partial derivatives of the cost function J with respect to each
    # parameter are:
    Theta1_grad <- Theta1_grad + Reg1
    Theta2_grad <- Theta2_grad + Reg2
    
    # Now we just "unroll" these into one long vector of parameters
    c(as.vector(Theta1_grad), as.vector(Theta2_grad))
}


# Let's just check this is working
compute_grad(init_params, Xtrain, ytrain, 1)

# Now we have everything we need to train the network!



#===============================================================================
# Section 7: Training the network ----
#===============================================================================

# Let's set lambda here so we can adjust it later
lambda <- 1

# We train the network by minimising the cost function - which we achieve by
# adjusting the network's parameters. We use R's "optim" function, and some
# "anonymous functions" (basically, compute_cost and compute_grad with 3 of the
# inputs fixed) as arguments. The control paramater "maxit" is the maximum
# number of iterations (or "training epochs") to perform before returning an
# answer.

optim_out <- optim(init_params,
                   function(x) compute_cost(x, Xtrain, ytrain, lambda),
                   function(x) compute_grad(x, Xtrain, ytrain, lambda),
                   method = "L-BFGS-B",
                   control = list(maxit = 50))

# optim returns several outputs in a list. The first is the optimised
# parameters:
nn_params <- optim_out[[1]]

# (The second output is the minimised cost)
optim_out[[2]]


# So now we can make predictions!
preds <- predict(nn_params, Xval)

# How well did we do? We assess performance with the cross-validation set
sprintf("Accuracy: %.1f%%", sum(preds == yval) / length(yval) * 100)


# We can have a look at what features the network has learned to look for
Theta1 <- make_thetas(nn_params)[[1]]

visualise(Theta1[1, ])
visualise(Theta1[16, ])



#===============================================================================
# Section 8: Improving performance ----
#===============================================================================

# We can adjust lambda and adjust the number of neurons in the hidden layer -
# these are the network's "hyper-parameters"

# Generally this is a matter of trial and error!

# We'll set a new value of lambda, which might reduce overfitting
lambda <- 4

# And we can change the size of the hidden layer - more neurons means the
# network learns to look for more features, which improves performance but also
# increases the likelihood of overfitting (and it's more computationally
# expensive, due to larger matrices!)
hidden_size <- 50

n1 <- 785 * hidden_size
n2 <- (hidden_size + 1) * 10
init_params <- runif((n1+n2), min = -epsilon, max = epsilon)

# We need to calculate a new set of init_params (because we don't have enough
# now that we've made the net bigger). We can just run the earlier code again
# (or if we're going to keep changing the structure of the network it might be
# worth writing a short function).

# Now we can re-train and re-evaluate using the test set (which hasn't
# been seen by the network yet - so the net can't overfit it!)

optim_out <- optim(init_params,
                   function(x) compute_cost(x, Xtrain, ytrain, lambda),
                   function(x) compute_grad(x, Xtrain, ytrain, lambda),
                   method = "L-BFGS-B",
                   control = list(maxit = 50))

nn_params <- optim_out[[1]]

preds <- predict(nn_params, Xtest)

sprintf("Accuracy: %.1f%%", mean(preds == ytest) * 100)



#===============================================================================
# Section 9: Kaggle submission ----
#===============================================================================

# We'll read in the test dataset from Kaggle
test <- read.csv("C:/Users/Owen/Desktop/nerualnet/data/test.csv")

# We have to do the same transformations as we did for the training set
Test <- as.matrix(test)
Test <- Test / max(Test)

# Let's make our predictions!
preds <- predict(nn_params, Test)

# Remember to replace "10" with "0" again
preds[preds == 10] <- 0

# Reformat into a data frame, ready to be written to a CSV
final <- data.frame(ImageId = c(1:28000), Label = preds)

# Write a CSV
write.csv(final, file = "~/Coding/DigitRecognizer/data/submission.csv",
          row.names = FALSE)


# URL for submission:
"https://www.kaggle.com/c/digit-recognizer/submissions/attach"



#===============================================================================
#===============================================================================


