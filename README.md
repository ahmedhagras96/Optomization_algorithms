# Optomization_algorithms
### Newton-Raphson Method
- In numerical analysis, Newton's method, also known as the Newton–Raphson method, named after Isaac Newton and Joseph Raphson, is a root-finding algorithm which produces successively better approximations to the roots of a real-valued function.
- Loosely speaking, it is a method for approximating solutions to equations. 
- for more information about it : https://en.wikipedia.org/wiki/Newton%27s_method

### Gradient descent
- So far, we used Newton–Raphson method to solve 1D function (function of one variable 𝒇(𝒙)). Now, we want to generalize this concept in order to find the steepest way towards the minima of a multidimensional function (function with multiple variables 𝒇(𝒙_𝟏,𝒙_𝟐,……,𝒙_𝒏)). Which is always the case in DS problems.
This is the where the Gradient Descent comes in 
- Gradient Descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. 
- The idea is to take repeated steps in the opposite direction of the gradient of the function at the current point, because this is the direction of steepest descent.
-  This algorithm and its variants have been proven effective to solve data related problems, especially in the  domain of neural networks. It’s not the only algorithm or the best but it is seen as the « hello world » of data science.
- for more information: https://en.wikipedia.org/wiki/Gradient_descent

### Batch/Vanilla GD Reminder
- Batch (vanilla) gradient descent, computes the gradient of the cost function w.r.t. the parameters θ for the entire training dataset:
![image](https://user-images.githubusercontent.com/32541520/155439088-eed23baf-8b22-4409-bcfd-5a7f957ebccc.png)
- Standard Gradient descent updates the parameters only after each epoch i.e. after calculating the derivatives for all the observations it updates the parameters. This phenomenon may lead to the following problems:
- It can be very slow for very large datasets because only one-time update for each epoch. Large number of epochs is required to have a substantial number of updates.
- For large datasets, the vectorization of data doesn’t fit into memory.
- For non-convex surfaces, it may only find the local minimums.

### Stochastic GD (SGD)
- Stochastic gradient descent updates the parameters for each observation which leads to more number of updates.
![stocastic](https://user-images.githubusercontent.com/32541520/155439631-c9a7dcd0-5acb-4e0a-a0c6-00b444e4d835.png)
- Disadvantages of SGD:
- Due to frequent fluctuations, it will keep overshooting near to the desired exact minima.
Add noise to the learning process i.e. the variance becomes large since we only use 1 example for each learning step.
- Increase run time.
- We can’t utilize vectorization over 1 example
- change1

### Mini Batch GD 
- Instead of going over all examples, Mini-batch Gradient Descent sums up over lower number of examples based on the batch size. Therefore, learning happens on each mini-batch of b examples:
![mine batch](https://user-images.githubusercontent.com/32541520/155439850-7337ea2c-ee7f-4fed-a0a1-3e5a5adc8f87.png)
- Advantages of Mini-batch GD:
- Updates are less noisy compared to SGD which leads to better convergence.
- A high number of updates in a single epoch compared to GD so less number of epochs are required for large datasets.
- Fits very well to the processor memory which makes computing faster.
- Note: The batch size is something we can tune. It is usually chosen as power of 2 such as 32, 64, 128, 256, 512, etc.

### Momentum-based GD
- Consider a case where in order to reach to your desired destination you are continuously being asked to follow the same direction and once you become confident that you are following the right direction then you start taking bigger steps and you keep getting momentum in that same direction.
- Similar to this if the gradient is in a flat surface for long term then rather than taking constant steps it should take bigger steps and keep the momentum continue. This approach is known as momentum based gradient descent.

### Nesterov Accelerated  GD (NAG)
- This looking ahead helps NAG in finishing its job (finding the minima) quicker than momentum-based GD. Hence the oscillations are less compared to momentum based GD and also there are fewer chances of missing the minima

### Adagrad
- Adagrad adopts the learning rate(η) based on the sparsity of features. So, the parameters with small updates (sparse features) have high learning rate whereas the parameters with large updates (dense features) have low learning rate. 
- v(t) accumulates the running sum of square of the gradients. Square of ∇w(t) neglects the sign of gradients. 
- v(t) indicates accumulated gradient up to time t.
- Epsilon (E) in the denominator avoids the chances of divide by zero error.
- if v(t) is low (due to less update up to time t) for a parameter then the effective learning rate will be high and if v(t) is high for a parameter then effective learning rate will be less.
![image](https://user-images.githubusercontent.com/32541520/155440348-bd84f8df-ea18-44a9-93bb-d854f44203bc.png)
- Advantage: parameters corresponding to sparse features get better updates.
- Disadvantage: the learning rate decay very aggressively as the denominator grows (not good for parameter corresponding to dense feature) hence there is no update in value of parameter so learning rate gets killed because denominator growing very fast. it reaches to near the minima point but not at the minima.

### RMSProp
- RMSProp Overcomes the decaying learning rate problem of adagrad and prevents the rapid growth in v(t).
- Instead of accumulating squared gradients from the beginning, it accumulates the previous gradients in some portion(weight).
- v(t) is exponentially decaying average of all the previous squared gradients. 
- Prevents rapid growth of v(t).
- The algorithm keeps learning and tries to converge.
![image](https://user-images.githubusercontent.com/32541520/155440643-e32223dd-da4d-4e7c-abc7-bc135763f737.png)
- RMSProp concerns with adaptive learning rate. However, it suffers from a large number of oscillations with high learning rate or large gradient.

### Adam
- Adaptive Moment Estimation (Adam) computes the exponentially decaying average of previous gradients m(t) along with an adaptive learning rate. 
- Adam is a combined form of Momentum-based GD and RMSProp.
- In Momentum-based GD, previous gradients(history) are used to compute the current gradient whereas, in RMSProp previous gradients(history) are used to adjust the learning rate based on the features. 
![image](https://user-images.githubusercontent.com/32541520/155440932-c0eec566-9cab-418b-896d-c213b271c222.png)
Traditionally β1 = 0.9, β2 = 0.999, and ε = 1e-8
η can work fine for the values 0.0001 and 0.001  




