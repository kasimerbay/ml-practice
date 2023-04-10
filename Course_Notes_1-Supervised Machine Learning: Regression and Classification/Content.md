
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

  ### **Lineer Regression**:
   We have _feature_ $x^{(i)}$ variable and _target_ $y^{(i)}$ variable.

   The process looks like the following;  
   a. **Training Set**  
   b. **Learning Algorithm**  
   c. $f_{w,b}(x^{(i)})$: Model that takes feature _x_ as input and returns prediction $\hat{y}^{(i)}$.
   d. Minimize the Cost Function given below to fit the model as good as possible.

   → How do we build a function with parameters(weights or coefficients) $w$ and $b$ so that $\hat{y}$ is close to the other _(feature, target)_ values?

    * The answer is the **Cost Function**

    $$J(w,b) = \frac{1}{2m}\sum_{i = 1}^{m}(\hat{y}^{(i)} - y^{(i)})^2$$  

    where $(i)$ is the index for training data and $m$ is the number of training inputs.

    * Below we have the implementation of our _Cost Function_;
      ```Python
      def compute_cost(x, y, w, b):
        """
        Computes the cost function for linear regression.

        Args:
            x (ndarray): Shape (m,) Input to the model (Population of cities)
            y (ndarray): Shape (m,) Label (Actual profits for the cities)
            w, b (scalar): Parameters of the model

        Returns
            total_cost (float): The cost of using w,b as the parameters for linear regression
                   to fit the data points in x and y
        """
        # number of training examples
        m = x.shape[0]

        # You need to return this variable correctly
        total_cost = 0

        ### The Main Idea Starts Here ###  
        for i in range(m):
            f_wb = w*x[i] + b
            loss = (f_wb - y[i])**2
            total_cost += loss
        total_cost = (1/(2*m)) * total_cost
        ### Ends Here ###

        return total_cost
      ```
      We call our function as below;
      ```Python
      # Compute cost with some initial values for paramaters w, b
      initial_w = 2
      initial_b = 1

      cost = compute_cost(x_train, y_train, initial_w, initial_b)
      print(f'Cost at initial w: {cost:.3f}')
      ```
      and it returns;
      ```Python
      » Cost at initial w: 75.203
      ```

    * To find the best values of $w$ and $b$ we have **Gradient Descent Algorithm**

      <ins>Steps of the **Gradient Descent**</ins>:

      1. Set your $w$ as follows &rarr; $w - \alpha \frac{d J(w,b)}{d w}$ to step away your starting $w$;

        $$\frac{\partial J(w,b)}{\partial w} = \frac{1}{m}\sum_{i = 1}^{m}(\hat{y}^{(i)} - y^{(i)}).\hat{x}^{(i)}$$
        where $\hat{y}^{(i)} = f_{w,b}(x^{(i)})$

        Also set your $b$ as follows &rarr; $b - \alpha \frac{d J(w,b)}{d b}$ to step away your starting $b$.

        $$\frac{\partial J(w,b)}{\partial b} = \frac{1}{m}\sum_{i = 1}^{m}(\hat{y}^{(i)} - y^{(i)})$$  
        where $\hat{y}^{(i)} = f_{w,b}(x^{(i)})$

        ```Python
        def compute_gradient(x, y, w, b):
            """
            Computes the gradient for linear regression
            Args:
              x (ndarray): Shape (m,) Input to the model (Population of cities)
              y (ndarray): Shape (m,) Label (Actual profits for the cities)
              w, b (scalar): Parameters of the model  
            Returns
              dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
              dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
             """

            # Number of training examples
            m = x.shape[0]

            # You need to return the following variables correctly
            dj_dw = 0
            dj_db = 0

            ### START CODE HERE ###
            for i in range(m):

                f_wb = w*x[i] + b

                dj_dw_i = (f_wb - y[i])*x[i]
                dj_db_i = f_wb - y[i]

                dj_dw += dj_dw_i
                dj_db += dj_db_i

            dj_dw *= 1/m
            dj_db *= 1/m
            ### END CODE HERE ###

            return dj_dw, dj_db
        ```
        We call our function as follows;
        ```Python
        # Compute and display gradient with w initialized to zeroes
        initial_w = 0
        initial_b = 0

        tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
        print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)
        ```

        Here $\alpha$ is *the learning rate* and the derivative gives which direction the descent step will be taken.

      2. Choose your learning rate $\alpha$

      We now finalize the algorithm.

      ```Python
      def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
          """
          Performs batch gradient descent to learn theta. Updates theta by taking
          num_iters gradient steps with learning rate alpha

          Args:
            x :    (ndarray): Shape (m,)
            y :    (ndarray): Shape (m,)
            w_in, b_in : (scalar) Initial values of parameters of the model
            cost_function: function to compute cost
            gradient_function: function to compute the gradient
            alpha : (float) Learning rate
            num_iters : (int) number of iterations to run gradient descent
          Returns
            w : (ndarray): Shape (1,) Updated values of parameters of the model after
                running gradient descent
            b : (scalar)                Updated value of parameter of the model after
                running gradient descent
          """

          # number of training examples
          m = len(x)

          # An array to store cost J and w's at each iteration — primarily for graphing later
          J_history = []
          w_history = []
          w = copy.deepcopy(w_in)  #avoid modifying global w within function
          b = b_in

          for i in range(num_iters):

              # Calculate the gradient and update the parameters
              dj_dw, dj_db = gradient_function(x, y, w, b )  

              # Update Parameters using w, b, alpha and gradient
              w = w - alpha * dj_dw               
              b = b - alpha * dj_db               

              # Save cost J at each iteration
              if i<100000:      # prevent resource exhaustion
                  cost =  cost_function(x, y, w, b)
                  J_history.append(cost)

              # Print cost every at intervals 10 times or as many iterations if < 10
              if i% math.ceil(num_iters/10) == 0:
                  w_history.append(w)
                  print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

          return w, b, J_history, w_history #return w and J,w history for graphing
      ```

      Everything above is covered in the following [Jupyter Notebook](https://github.com/kasimerbay/ml-practice/blob/master/Lineer%20Regression%20Notes/Lineer%20Regression.ipynb)

      ### Lineer Regression with Multiple Features  
      Multiple fetaures implies out feature $x^{(i)}$ is a raw vector that is we give feature to our model as a vector; say $\vec{x}_{j}$ where $x_{j}$ is the $x_{j}$ feature.

      So our models becomes;  

      $$ f_{\vec{w},b}({\vec{x}}) = [w_{1}\ w_{2}\ \cdots\ w_{n}] \cdot [x_{1}\  x_{2}\  \cdots\ x_{n}] + b = w_{1}x_{1} + w_{2}x_{2} + \cdots + w_{n}x_{n} + b $$


      1. <ins>**Vectorization**</ins>  
      In regression, vectorization makes your code shorter and also makes it run much more efficient. We vectorize arrays using [Numpy](https://numpy.org/doc/stable/).

      2. <ins>**Gradient Descent for Multiple Lineer Regression**</ins>

        $$ \mbox{repeat until convergence:} \lbrace$$

        $$ w_j = w_j -  \alpha \frac{\partial J(\vec{w},b)}{\partial w_j}$$

        $$ b\ \ = b -  \alpha \frac{\partial J(\vec{w},b)}{\partial b} \rbrace$$

        $$ \mbox{for j = 0..n-1} $$  

        where, n is the number of features, parameters $w_j$,  $b$, are updated simultaneously and where  

        $$\begin{align}
        \frac{\partial J(\vec{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})\vec{x}^{(i)}_{j}  \\
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

      ---

      You can reach the Lineer Regression Notebook [here](https://github.com/kasimerbay/ml-practice/blob/master/Lineer%20Regression%20Notes/Lineer%20Regression.ipynb)

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
