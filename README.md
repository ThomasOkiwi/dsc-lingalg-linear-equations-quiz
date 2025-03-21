
# Linear Algebra and Calculus - Introduction
  
### Introduction
In this section, we're going to take a step back to learn some of the basics of the math that powers the most popular machine learning models. You may not need deep knowledge of linear algebra and calculus just to build a scikit-learn model, but this introduction should give you a better understanding of how your models are working "under the hood".
## Linear Algebra
Linear Algebra is so foundational to machine learning that you're going to see it referenced many times as the course progresses. In this section, the goal is to give you both a theoretical introduction and some computational practice, solving a real-life problem by writing the code required to solve a linear regression using OLS.
We're going to kick this section off by looking at some of the many places that linear algebra is used in machine learning - from deep learning to natural language processing (NLP) to dimensionality reduction techniques such as principal component analysis (PCA).
## Systems of Linear Equations
We then start to dig into the math! We look at the idea of linear simultaneous equations - a set of two or more equations each of which is linear (can be plotted on a graph as a straight line). We then see how such equations can be represented as vectors or matrices to represent such systems efficiently.
## Scalars, Vectors, Matrices, and Tensors
In a code along, we'll introduce the concepts and concrete representations (in NumPy) of scalars, vectors, matrices, and tensors - why they are important and how to create them.
## Vector/Matrix Operations
We then start to build up the basic skills required to perform matrix operations such as addition and multiplication. You will also cover key techniques used by many machine learning models to perform their calculations covering both the Hadamard product and the (more common) dot product.
## Solving Systems of Linear Equations Using NumPy
We then bring the previous work together to look at how to use NumPy to solve systems of linear equations, introducing the identity and inverse matrices along the way.
## Regression Analysis Using Linear Algebra and NumPy
Having built up a basic mathematical and computational foundation for linear algebra, you will solve a real data problem - looking at how to use NumPy to solve a linear regression using the ordinary least squares (OLS) method.
Computational Complexity
In the last linear algebra lesson, we look at the idea of computational complexity and Big O notation, showing why OLS is computationally inefficient, and that a gradient descent algorithm can instead be used to solve a linear regression much more efficiently.
## Calculus and Gradient Descent
Next, you'll learn about the mechanism behind many machine learning optimization algorithms: gradient descent! Along the way, we'll also look at cost functions and will provide a foundation in calculus that will be valuable to you throughout your career as a data scientist.
Just as we used solving a linear regression using OLS as an excuse to introduce you to linear algebra, we're now using the idea of gradient descent to introduce enough calculus to both understand and have good intuitions about many of the machine learning models that you're going to learn throughout the rest of the course.
## An Introduction to Calculus and Derivatives
We're going to start off by introducing derivatives - the "instantaneous rate of change of a function" or (more graphically) the "slope of a curve". We'll start off by looking at how to calculate the slope of a curve for a straight line, and then we'll explore how to calculate the rate of change for more complex (non-linear) functions.
Gradient Descent
Now that we know how to calculate the slope of a curve - and, by extension, to find a local minimum (low point) or maximum (high point) where the curve is flat (the slope of the curve is zero), we'll look at the idea of a gradient descent to step from some random point on a cost curve to find the local optima to solve for a given linear equation. We'll also look at how best to select the step sizes for descending the cost function, and how to use partial derivatives to optimize both slope and offset to more effectively solve a linear regression using gradient descent.

# Motivation for Linear Algebra in Data Science
  
## Introduction
In this section, you'll learn about algebra as a foundational step for data science, and later on statistics. Linear algebra is also very important when moving on to machine learning models, where a solid understanding of linear equations plays a major role. This lesson will attempt to present some motivational examples of how and why a solid foundation of linear algebra is valuable for data scientists.
## Objectives
You will be able to:
•	State the importance of linear algebra in the fields of data science and machine learning
•	Describe the areas in AI and machine learning where linear algebra might be used for advanced analytics
## Background
While having a deep understanding of linear algebra may not be mandatory, some basic knowledge is undoubtedly extremely helpful in your journey towards becoming a data scientist.
You may already know a number of linear algebraic concepts without even knowing it. Examples are: matrix multiplication and dot-products. Later on, you'll learn more complex algebraic concepts like the calculation of matrix determinants, cross-products, and eigenvalues/eigenvectors. As a data scientist, it is important to know some of the theories as well as having a practical understanding of these concepts in a real-world setting.
## An analogy
Think of a simple example where you first learn about a sine function as an infinite polynomial while learning trigonometry. Students usually practice this function by passing different values to this function and getting the expected results and then manage to relate this to triangles and vertices. When learning advanced physics, students get to learn more applications of sine and other similar functions in the area of sound and light. In the domain of Signal Processing for unidimensional data, these functions pop up again to help you solve filtering, time-series related problems. An introduction to numeric computation around sine functions can not alone help you understand its wider application areas. In fact, sine functions are everywhere in the universe from music to light/sound/radio waves, from pendulum oscillations to alternating current.
##  Why Linear Algebra?
Linear algebra is the branch of mathematics concerning vector spaces and linear relationships between such spaces. It includes the study of lines, planes, and subspaces, but is also concerned with properties common to all vector spaces.
Analogous to the example we saw above, it's important that a data scientist understands how data structures are built with vectors and matrices following the geometric intuitions from linear algebra, in addition to the numeric calculations. A data-focused understanding of linear algebra can help machine learning practitioners decide what tools can be applied to a given problem and how to interpret the results of experiments. You'll see that a good understanding of linear algebra is particularly useful in many ML/AI algorithms, especially in deep learning, where a lot of the operations happen under the hood.
Following are some of the areas where linear algebra is commonly practiced in the domain of data science and machine learning:
## Computer Vision / Image Processing
 ![image](https://github.com/user-attachments/assets/344332f5-c66d-44b9-af8d-26b33bd3ceeb)

Computers are designed to process binary information only (only 0s and 1s). How can an image such as the dog shown here, with multiple attributes like color, be stored in a computer? This is achieved by storing the pixel intensities for red, blue and green colors in a matrix format. Color intensities can be coded into this matrix and can be processed further for analysis and other tasks. Any operation performed on this image would likely use some form of linear algebra with matrices as the back end.
## Deep Learning - Tensors
Deep Learning is a sub-domain of machine learning, concerned with algorithms that can imitate the functions and structure of a biological brain as a computational algorithm. These are called artificial neural networks (ANNs).
The algorithms usually store and process data in the form of mathematical entities called tensors. A tensor is often thought of as a generalized matrix. That is, it could be a 1-D matrix (a vector is actually such a tensor), a 2-D matrix (like a data frame), a 3-D matrix (something like a cube of numbers), even a 0-D matrix (a single number), or a higher dimensional structure that is harder to visualize.
 ![image](https://github.com/user-attachments/assets/01278464-4696-4e7f-bd92-d3d29255a141)

As shown in the image above where different input features are being extracted and stored as spatial locations inside a tensor which appears as a cube. A tensor encapsulates the scalar, vector, and the matrix characteristics. For deep learning, creating and processing tensors and operations that are performed on these also require knowledge of linear algebra. Don't worry if you don't fully understand this right now, you'll learn more about tensors later!
## Natural Language Processing
Natural Language Processing (NLP) is another (very popular) area in Machine Learning dealing with text data. The most common techniques employed in NLP include BoW (Bag of Words) representation, Term Document Matrix etc. As shown in the image below, the idea is that words are being encoded as numbers and stored in a matrix format. Here, we just use 3 sentences to illustrate this:

![image](https://github.com/user-attachments/assets/85676f9f-e491-4ec6-a6e2-0f2dfc1ba836)

 
This is just a short example, but you can store long documents in (giant) matrices like this. Using these counts in a matrix form can help perform tasks like semantic analysis, language translation, language generation etc.
## Dimensionality Reduction
Dimensionality reduction techniques, which are heavily used when dealing with big datasets, use matrices to process data in order to reduce its dimensions. Principle Component Analysis (PCA) is a widely used dimensionality reduction technique that relies solely on calculating eigenvectors and eigenvalues to identify principal components as a set of highly reduced dimensions. The picture below is an example of a three-dimensional data being mapped into two dimensions using matrix manipulations.
 ![image](https://github.com/user-attachments/assets/0ac92a03-c1cb-4db6-bd1f-bdae1c1ad4dc)

Great, you now know about some key areas where linear algebra is used! In the following lessons, you'll go through an introductory series of lessons and labs that will cover basic ideas of linear algebra: an understanding of vectors and matrices with some basic operations that can be performed on these mathematical entities. We will implement these ideas in Python, in an attempt to give you the foundational knowledge to deal with these algebraic entities and their properties. These skills will be applied in advanced machine learning sections later in the course.
## Further Reading


## Systems of Linear Equations
  
### Introduction
Linear algebra is a sub-field of mathematics concerned with vectors, matrices, and linear transforms between them. The first step towards developing a good understanding of linear algebra is to get a good sense of what linear mappings and linear equations are, how these relate to vectors and matrices and what this has to do with data analysis. Let's try to develop a basic intuition around these ideas by first understanding what linear equations are.
## Objectives
You will be able to:
•	Describe a system of linear equations for solving analytical problems
•	Describe how matrices and vectors can be used to solve linear equations
•	Solve a system of equations using elimination and substitution
What are linear equations?
In mathematics, a system of linear equations (or linear system) is a collection of two or more linear equations involving the same set of variables. For example, look at the following equations:
3x+2y−z=0
2x−2y+4z=−2
−x+0.5y−z=0
This is a system of three equations in the three variables x, y, and z. A solution to a linear system is an assignment of values to the variables in a way that all the equations are simultaneously satisfied. A solution to the system above is given by:
x=1
y=−8/3
z=−7/3
These values make all three equations valid. The word "system" indicates that the equations are to be considered collectively, rather than individually.
## Solving linear equations
A system of linear equations can always be expressed in a matrix form. Algebraically, both of these express the same thing. Let's work with an example to see how this works:
#### Example
Let's say you go to a market and buy 2 apples and 1 banana. For this, you end up paying 35 pence. If you denote apples by a and bananas by b, the relationship between items bought and the price paid can be written down as an equation - let's call it Eq. A:
2a+b=35 - (Eq. A)
On your next trip to the market, you buy 3 apples and 4 bananas, and the cost is 65 pence. Just like above, this can be written as Eq. B:
3a+4b=65 - (Eq. B)
These two equations (known as a simultaneous equations) form a system that can be solved by hand for values of a and b i.e., price of a single apple and banana.
Let's solve this system for individual prices using a series of eliminations and substitutions:
<b>Step 1</b>: Multiply Eq. A by 4
8a+4b=140 - (Eq. C)
<b>Step 2 1</b>: Subtract Eq. B from Eq. C
5a=75 which leads to a=15
<b>Step 31</b>: Substitute the value of a in Eq. A
30+b=35 which leads to b=5
So the price of an apple is 15 pence and the price of the banana is 5 pence.
From equations to vectors and matrices
Now, as your number of shopping trips increase along with the number of items you buy at each trip, the system of equations will become more complex and solving a system for individual price may become very expensive in terms of time and effort. In these cases, you can use a computer to find the solution.
The above example is a classic linear algebra problem. The numbers 2 and 1 from Eq. A and 3 and 4 from Eq. B are linear coefficients that relate input variables a and b to the known output 15 and 5.
Using linear algebra, we can write this system of equations as shown below:
 ![image](https://github.com/user-attachments/assets/8c30a231-4d30-45ed-b857-17a2d4edd248)

You see that in order for a computational algorithm to solve this (and other similar) problems, we need to first convert the data we have into a set of matrix and vector objects. Machine learning involves building up these objects from the given data, understanding their relationships and how to process them for a particular problem.
Solving these equations requires knowledge of defining these vectors and matrices in a computational environment and of operations that can be performed on these entities to solve for unknown variables as we saw above. We'll look into how to do this in upcoming lessons.




# Systems of Linear Equations - Lab

## Introduction
The following scenarios present problems that can be solved as a system of equations while performing substitutions and eliminations as you saw in the previous lesson.

* Solve these problems by hand, showing all the steps to work out the unknown variable values 
* Verify your answers by showing the calculated values satisfy all equations

## Objectives

In this lab you will: 

- Solve a system of equations using elimination and substitution

## Exercise 1
Jane paid 12 dollars for 4 cups of coffee and 4 cups of tea. 3 cups of coffee cost as much as 2 cups of tea. What would be the total cost of 5 cups of coffee and 5 cups of tea?

### Solution

> Let $x$ be the unit price of coffee and $y$ be the unit price of tea


```python
# Your solution here 
# Answer: 5 cups of tea and 5 cups of coffee = 15 dollars
```

## Exercise 2

Jim has more money than Bob. If Jim gave Bob 20 dollars, they would have the same amount. If Bob gave Jim 22 dollars, however, Jim would then have twice as much as Bob. 

How much does each one actually have?

### Solution
> Let x be the amount of money that Jim has and y be the amount that Bob has 


```python
# Your solution here 
# Answer:
# y = 106 (Bob's amount)
# x = 146 (Jim's amount)
```

## Exercise 3

Mia has 30 coins, consisting of quarters (25 cents) and dimes (10 cents), which totals to the amount 5.70 dollars.  
How many of each does she have?

### Solution

> Let x be the number of quarters and y be the number of dimes 


```python
# Your solution here 
# Answer:
# x = 18 quarters
# y = 12 dimes
```

## Level up (Optional)
For more practice with linear equations, visit the following links for more complex equations:

* https://www.transum.org/software/SW/Starter_of_the_day/Students/Simultaneous_Equations.asp?Level=6
* https://www.transum.org/software/SW/Starter_of_the_day/Students/Simultaneous_Equations.asp?Level=7

## Summary

In this lesson, you learned how to solve linear equations by hand to find the coefficient values. You'll now move forward to have a deeper look into vectors and matrices and how Python and NumPy can help us solve more complex equations in an analytical context. 
