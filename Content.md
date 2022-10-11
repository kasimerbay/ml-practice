<ins>**Dedicated to my beloved [Gülçehrem](ahmetkasimerbay.net)**</ins>

- Machine Learning examples,
    - Image recognition,
    - recomendation engines,
    - spam detection,
    - fighting against climate changing by locating best places for wind turbines,
    - help doctors making good diagnoses.

---

# Week 1

### What is Machine Learning?

> Field of study that gives computers the ability to learn without explicitly programmed.
Arthur Samuel (1959)
>

## Supervised Learning

This is the type that you give the machine to learn from “the right answers”.  Supervised Learning learn from  data labeled with “right answer”

- Spam Filtering
- Speech Recognition
- Machine Translation
- Online Advertising
- Self-Driving Car
- Visual Inspection

### Regression Models
→**Regression** is a method of Supervised Learning

- Estimating rental prices for houses from a given sample of houses and price pairs.  

1. **Lineer Regression**:
   We have _feature_ ($x^{(i)}$) variable and _target_ ($y^{(i)}$) variable.

   The process looks like the following;  
   a. **Training Set**  
   b. **Learning Algorithm**  
   c. $f_{w,b}(x^{(i)})$: Model that takes feature _x_ as input and returns prediction $\hat{y}^{(i)}$.
   d. Minimize the Cost Function given below to fit the model as good as possible.

   → How do we build a function with parameters(weights or coefficients) $w$ and $b$ so that $\hat{y}$ is close to the other _(feature, target)_ values?

   * The answer is the **Cost Function**

    $$J(w,b) = \frac{1}{2m}\sum_{i = 1}^{m}(\hat{y}^{(i)} - y^{(i)})^2$$  

    where _(i)_ is the index for training data and _m_ is the number of training inputs.

    * To find the best values of $w$ and $b$ we have **Gradient Descent Algorithm**

      <ins>Steps of the **Gradient Descent**</ins>:

      1. Set your $w$ as follows &rarr; $w - \alpha \frac{d J(w,b)}{d w}$ to step away your starting $w$;

            $$ \frac{\partial J(w,b))}{\partial w} = \frac{1}{m}\sum_{i = 1}^{m}(\hat{y}^{(i)} - y^{(i)}).\hat{x}^{(i)} $$  
            where $\hat{y}^{(i)} = f_{w,b}(x^{(i)})$

            Also set your $b$ as follows &rarr; $b - \alpha \frac{d J(w,b)}{d b}$ to step away your starting $b$.

            $$ \frac{\partial J(w,b))}{\partial b} = \frac{1}{m}\sum_{i = 1}^{m}(\hat{y}^{(i)} - y^{(i)})$$  
            where $\hat{y}^{(i)} = f_{w,b}(x^{(i)})$

            Here $\alpha$ is *the learning rate* and the derivative gives which direction the descent step will be taken.

      2. Choose your learning rate $\alpha$  

→ **Classification** is another method for Supervised Learning

- Classification predict categories (classes) like in,
    - cat-dog distinction,
    - malignant-benign
    - 0,1,2… (finite number)
- This is different from Regression because its output has small number

## Unsupervised Learning

In this type of data we are not given any answers. We try to find something interesting in the given data.

- Used in Google News

→ **Clustering** is a method of Unsupervised Learning

- DNA Clustering
- Customer Clustering

→ Anomaly Detection

- Fraud Detection in financial system

---
# Week 2

### Lineer Regression with Multiple Features  
Multiple fetaures implies out feature ,$x^{(i)}$, is a raw vector that is we give feature to our model as a vector; say $\vec{x}_j$, where $\vec{x}_j$ is the $x_{j}$ feature.

So our models becomes;  

$$ f_{\vec{w},b}({\vec{x}}) = [w_{1}\ w_{2}\ \cdots\ w_{n}] \cdot [x_{1}\  x_{2}\  \cdots\ x_{n}] + b = w_{1}x_{1} + w_{2}x_{2} + \cdots + w_{n}x_{n} + b $$


1. <ins>**Vectorization**</ins>  
In regression, vectorization makes your code shorter and also makes it run much more efficient. We vectorize arrays using [Numpy](https://numpy.org/doc/stable/).

2. <ins>**Gradient Descent for Multiple Lineer Regression**</ins>  

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& w_j = w_j -  \alpha \frac{\partial J(w,b)}{\partial w_j} \; & \text{for j = 0..n-1}\newline
&b\ \ = b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}$$  
where, n is the number of features, parameters $w_j$,  $b$, are updated simultaneously and where  

$$\begin{align}
\frac{\partial J(\vec{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})\vec{x}_{j}^{(i)}  \\
\frac{\partial J(\vec{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})
\end{align}$$

* m is the number of training examples in the data set


*  $f_{\mathbf{\vec{w}},b}(\vec{x}^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target value

3. <ins>**Feature Scaling**</ins>  
  This enables us to run *Gradient Descent* much faster. In order to achieve scaling, we use some methods such as *diving all entries by the greatest entry*, *mean normalizaiton* and *Z-score normalizaiton*

  - Feature scaling, essentially dividing each positive feature by its maximum value, or more generally, rescale each feature by both its minimum and maximum values using $(x-min)/(max-min)$. Both ways normalizes features to the range of -1 and 1, where the former method works for positive features which is simple and serves well for the lecture's example, and the latter method works for any features.

  - Mean normalization: $x_i := \dfrac{x_i - \mu_i}{max - min}$

  - <ins>Z-score normalization</ins>:

    After z-score normalization, all features will have a mean of 0 and a standard deviation of 1.

    To implement z-score normalization, adjust your input values as shown in this formula:
$$x^{(i)}_j = \dfrac{x^{(i)}_j - \mu_j}{\sigma_j}$$
    where $j$ selects a feature or a column in the $X$ matrix. $µ_j$ is the mean of all the values for feature (j) and $\sigma_j$ is the standard deviation of feature (j).

$$
    \begin{align}
    \mu_j &= \frac{1}{m} \sum_{i=0}^{m-1} x^{(i)}_j\\
    \sigma^2_j &= \frac{1}{m} \sum_{i=0}^{m-1} (x^{(i)}_j - \mu_j)^2
    \end{align}
$$

  >**Implementation Note:** When normalizing the features, it is important
  to store the values used for normalization - the mean value and the standard deviation used for the computations. After learning the parameters
  from the model, we often want to predict the prices of houses we have not
  seen before. Given a new x value (living room area and number of bed-
  rooms), we must first normalize x using the mean and standard deviation
  that we had previously computed from the training set.

4. <ins>**How to Ensure Gradient Descent to Be Converged?**</ins>  
Try to visualize the *learning curve* and detect where the curve converges, or use the *automatic convergence test* by deciding an $\epsilon$ and if in the next iteration $J$ increases less than that $\epsilon$ than your $J$ is very close to converge. So the **Gradient Descent Algorithm** is about to find $(\vec{w},b)$ pairs such that $J$ in its global minimum.



5. <ins>**Choosing a Good Learning Rate**</ins>  
    * With a small enough $\alpha$, $J(\vec{w},b)$ should decrease on every iteration.

    * If $\alpha$ is too small, *Gradient Descent* takes a lot more iteration to converge.
