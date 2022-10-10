- Machine Learning examples,
    - Image recognition,
    - recomendation engines,
    - spam detection,
    - fighting against climate changing by locating best places for wind turbines,
    - help doctors making good diagnoses.

---

# Week 1

### What is Machine Learning?

> Field of study that gives computers the ability to learn without axplicitely programmed.
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

      <ins>Steps of the Gradient Descent</ins>:  

      1. Set your $w$ as follows &rarr; $w - \alpha \frac{d J(w,b)}{d w}$ to step away your starting $w$. Also set your $b$ as follows &rarr; $b - \alpha \frac{d J(w,b)}{d b}$ to step away your starting $b$. Here $\alpha$ is *the learning rate* and the derivative gives which direction the descent step will be taken.  

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
-
