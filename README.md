# Plotting intersection of curves using gradient descent

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/vdivakar/curves-intersection-with-gradient-descent/main/app.py#plotting-intersection-of-curves-using-gradient-descent)

<img src="https://github.com/vdivakar/curves-intersection-with-gradient-descent/blob/main/images/2cylinders.png" width=200 align=left>
<img src="https://github.com/vdivakar/curves-intersection-with-gradient-descent/blob/main/images/2doubleCones.png" width=200 align=left>
<img src="https://github.com/vdivakar/curves-intersection-with-gradient-descent/blob/main/images/doubleCone_cylinder.png" width=200>


This app is a result of finding a new application of gradient descent. I am a fan of plotting graphs and visualizations in general.
Hence, I was quite fascinated on discovering that we can plot graphs and their intersection using the concep of gradient descent.

Let's say you are giving equations of curves and you need to plot the intersection of these curves. As an example, say you have 2 spheres (3D), how would you plot the intersection of the given spheres? 

![](https://latex.codecogs.com/svg.latex?F_1(x)=|\textbf{x}-\textbf{a}|^2-R_1^2) 

![](https://latex.codecogs.com/svg.latex?F_2(x)=|\textbf{x}-\textbf{b}|^2-R_2^2)  ... **x**, **a** & **b** are vectors of size 3.



My first approach to this problem was finding the equation of intersection of these 2 functions by equating them i.e. F_1(x) = F_2(x).
Then trying to simply the equation and use that equation to plot the points. 
This approach is not feasible for 2 reasons:
1. Equating the 2 functions won't necessarily give you the equation of intersection. For instance, equating 2 equations of spheres will
give you intersection **plane** of the spheres and not the equation of intersecting circle (if any).
2. Even if you had an equation, the question still remains, how to plot points from a given equation?


If you observe, points that lie on the intersection of the curves should satisfy all the functions separately i.e. 
![](https://latex.codecogs.com/svg.latex?\forall{i}F_i(x)=0) 


So, another approach (highly ineffective) would be to generate points randomly everytime and see if they satisfy all the given equations.
If it does, it is a valid 'point'. Else, generate another random point and repeat untill you have sufficient points.
Downsides of this approach:
1. The search space is too big. Even bigger for N-dimensional points. 
2. Highly ineffective approach. Might take forever to stumble upon such valid points.

### Gradient Descent to the rescue

Can we modify the previous approach- Instead of discarding an invalid randomly generated point, can we update it iteratively so that it
approaches a valid solution? If so, what would it mean to be a valid solution and when should we stop updating the sample?

#### What should be the criteria for a point *x* to be a valid solution?

If the point ![](https://latex.codecogs.com/svg.latex?x^*) lies on the intersection of the curves, it should satisfy 
![](https://latex.codecogs.com/svg.latex?F_i(x^*)=0) for all *i* i.e. 

![](https://latex.codecogs.com/svg.latex?F_1(x^*)=0);  &

![](https://latex.codecogs.com/svg.latex?F_2(x^*)=0) 

We can define a function ![](https://latex.codecogs.com/svg.latex?G(x)) as the summation of the given functions ![](https://latex.codecogs.com/svg.latex?\sum_{}F_i(x)) to hold the above condition.

![](https://latex.codecogs.com/svg.latex?G(x)=F_1(x)+F_2(x))

So, we can say that a point will be valid when it satisfies G(x) = 0, since it will only hold when all the F_i(x) are zero. 
This will be our criterion for checking if the point is a valid solution.

However, we are not yet done. The range of G(x) can be from ![](https://latex.codecogs.com/svg.latex?(-\infty,\infty)). This means,
the minimum value of G(x) is not necessarily 0. This is a problem because if we try to minimize G(x) with gradient descent, it's highly
possible that it might attain some negative value as the minima.

So, we need to do slight modification in G(x) such that its minimum value is 0.

My first instict was to define G(x) as the sum of absolute F_i(x) i.e.

![](https://latex.codecogs.com/svg.latex?G(x)=|F_1(x)|+|F_2(x)|)

The minimum value of this function will be 0 and will hold all the conditions discussed above.
However, if are trying to use Gradient Descent, using modulus operation can be problematic because the function may not remain smooth anymore.

So, what's an easy alternative for modulus operator which also holds the smoothness property? - Use squares!

![](https://latex.codecogs.com/svg.latex?G(x)=(F_1(x))^2+(F_2(x))^2)

This function can now be minimised to get the points of intersection of the curves.
1. The function will be smooth and continuos. Provided F(x) are themselves smooth and continuous.
2. The minimum value of G(x) is zero.
3. The minimum value of G(x) represents the interesection of all F_i(x)

```
 Generate a random point x
 While G(x) != 0:
    x = x - lr * gradient(G(x))
    
 Repeat for N points.
```



<br>
<br>

Assumptions:
1. Curves do intersect somewhere.
2. The individual curves are themselves differentiable.
















