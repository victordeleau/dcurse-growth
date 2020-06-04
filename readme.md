# What is the sparsity growth rate of random points in a hypercube as its dimensionality increases ?

Exact solutions are known for the square and the cube (whose average distance between random points is known as Robbins constant = 0.6617..). For d-dimensional hypercubes, those distances can be estimated by Monte Carlo experiments.

This begs the question: how fast is this average distance growing? In machine learning, the curse of dimensionality informs us that easy to compute solutions in low dimensional space will quickly become computationally prohibitive as the dimensionality of the input space increases. Data points are usually considered to be i.i.d. In that case, if we add a dimension to our input space, how much more data points, on average, do we need for those data points to be as far away from each other as before?

The average distance between random points is computed up to a specific number of dimensions. A quick look at the data indicates that the growth rate is slowing as the dimensionality increases. Therefore, we take the log base e of every value on the X axis, and do a regular linear regression using the vectorized closed-form solution. We obtained a log curve with the following equation:

**y = 1.11 ln(x) - 1.22**

We used 1000000 samples per hypercube and dimensions from 1 to 128. The coefficient of determination R² was equal to 0.999673