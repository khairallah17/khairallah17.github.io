---
layout: post
title: "Linear Regression from Scratch"
categories: ["Machine Learning", "Mathematics"]
math: true
---

## 1. Problem

You just filled up your tank. But what's the actual cost going to look like as your car ages?

As a car accumulates kilometers, its fuel efficiency changes and so does what you pay at the pump. But can you actually predict the diesel cost from mileage alone?

Turns out, yes. And the math behind it is simpler than you'd think.

I decided to take a dataset of cars their mileage and diesel costs and build a model from scratch that learns the relationship between the two. No ML libraries, no shortcuts. Just raw math and Python.

This is what I learned along the way.

---

## 2. The Math Behind Linear Regression

Linear regression boils down to one simple idea: **draw the best possible straight line through your data**.

We assume the relationship between mileage and diesel cost is linear meaning as one goes up, the other changes at a constant rate. That line is described by:

$$
\hat{y} = \theta_0 + \theta_1 \cdot x
$$

Where:
- $x$ is the input the car's mileage in km
- $\hat{y}$ is the predicted diesel cost
- $\theta_0$ is the **intercept** the baseline cost when mileage is zero
- $\theta_1$ is the **slope** how much the cost changes for each additional km

In code, every prediction is just one line:

```python
y_pred = theta0 + (theta1 * x_train)
```

### The Cost Function

To measure how well our line fits the data, we need a way to quantify the error. For each data point, the error is the difference between what the model predicted and what the actual value is called a **residual**:

$$
e_i = \hat{y}_i - y_i
$$

We square each residual to penalise large errors and avoid negatives cancelling out positives. Summing them all gives us the **Residual Sum of Squares (RSS)**:

$$
RSS = \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$$

Dividing by the number of samples $m$ gives the **Mean Squared Error (MSE)** the function we want to minimise:

$$
MSE = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$$

The goal of training is to find the values of $\theta_0$ and $\theta_1$ that make this number as small as possible.

![MSE Cost Surface](/assets/img/posts/linear-regression/cost_surface.png)
_The MSE cost surface as a function of θ₀ and θ₁ training is the process of finding the lowest point._

---

## 3. Gradient Descent

We can't just guess $\theta_0$ and $\theta_1$. Instead, we use an algorithm called **gradient descent** it starts at zero and nudges the parameters little by little in the direction that reduces the error.

Think of it like walking downhill in the dark. You can't see the bottom, but you can feel which direction is downward and take a small step. Repeat that enough times, and you'll reach the lowest point.

### The Partial Derivatives

To know which direction is "downhill," we compute the **gradient** of the MSE with respect to each parameter. These are the partial derivatives:

$$
\frac{\partial MSE}{\partial \theta_0} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)
$$

$$
\frac{\partial MSE}{\partial \theta_1} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot x_i
$$

These tell us the slope of the error surface at our current position. We then move in the **opposite direction** by subtracting a fraction of the gradient controlled by the **learning rate** $\alpha$:

$$
\theta_0 := \theta_0 - \alpha \cdot \frac{\partial MSE}{\partial \theta_0}
$$

$$
\theta_1 := \theta_1 - \alpha \cdot \frac{\partial MSE}{\partial \theta_1}
$$

This update is repeated for every iteration (epoch) until the parameters converge.

```python
def gradient_descent(x_train, y_train, iterations, lr=0.01):
    theta0 = 0
    theta1 = 0
    mse_history = []
    m = len(y_train)

    for _ in range(iterations):
        y_pred = theta0 + (theta1 * x_train)
        error = y_pred - y_train

        mse = np.mean(error**2) / m
        mse_history.append(mse)

        grad_theta0 = (2 / m) * np.sum(error)
        grad_theta1 = (2 / m) * np.sum(error * x_train)

        theta0 -= lr * grad_theta0
        theta1 -= lr * grad_theta1

    return theta0, theta1, mse_history
```

### The Role of the Learning Rate

The learning rate $\alpha$ is one of the most important hyperparameters:

- **Too large**: the updates overshoot the minimum the model diverges and never converges
- **Too small**: training works but takes many more iterations than necessary
- **Just right**: smooth, steady descent toward the minimum

![Learning Rate Comparison](/assets/img/posts/linear-regression/learning_rate.png)
_Effect of different learning rates on convergence speed and stability._

---

## 4. Data Normalisation

Before training, I ran into a problem: mileage values go up to 240,000 km. Feed numbers that large directly into gradient descent and the gradients explode the algorithm overshoots and never converges.

The reason is that large input values cause large partial derivatives, which in turn cause large parameter updates. The model becomes unstable.

The fix is **normalisation**. I used Z-score normalisation (also called standardisation), which transforms the data to have a mean of $0$ and a standard deviation of $1$:

$$
x_{norm} = \frac{x - \mu}{\sigma}
$$

Where:
- $\mu$ is the mean of $x$
- $\sigma$ is the standard deviation of $x$

```python
def normalize(x_train):
    mean = x_train.mean()
    std = x_train.std()
    return (x_train - mean) / std
```

After normalisation, all input values sit in a small, stable range typically between $-3$ and $3$ — making gradient descent behave predictably.

### What Does the Distribution Look Like?

To sanity check the normalised data, I plotted its density curve using the normal CDF:

```python
def density_curve(X):
    scores = stats.norm.cdf(X)
    plt.plot(scores)
    plt.show()
```

![Density Curve](/assets/img/posts/linear-regression/density_curve.png)
_Density curve of the normalised mileage values a smooth S-curve confirms the data is well-distributed._

This is one of those steps that feels like a technicality until you skip it and watch your training blow up.

---

## 5. Evaluation Metrics

Training a model is one thing. Knowing whether it's any good is another. I used three metrics that together paint the full picture.

### Residual Sum of Squares (RSS)

The raw total squared error across all predictions:

$$
RSS = \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$

```python
def RSS(y_pred, y_train):
    rss = (y_train - y_pred) ** 2
    return np.sum(rss)
```

### Root Mean Squared Error (RMSE)

RSS is hard to interpret on its own because it scales with the dataset size. RMSE brings it back to the same units as the target variable in this case, the cost of diesel:

$$
RMSE = \sqrt{\frac{RSS}{m}}
$$

```python
def RMSE(rss, size):
    return math.sqrt(rss / size)
```

If RMSE is 200, it means the model's predictions are off by an average of 200 units of cost. Concrete and interpretable.

### R² Coefficient of Determination

R² tells you how much of the variation in the target is explained by your model. It compares the model's error (RSS) against the error of a naive baseline always predicting the mean:

$$
TSS = \sum_{i=1}^{m} (y_i - \bar{y})^2
$$

$$
R^2 = 1 - \frac{RSS}{TSS}
$$

```python
def evaluate(y_pred, y_train, mse_history):
    rss = RSS(y_pred, y_train)
    tss = TSS(y_train, np.mean(y_train))
    r_square = 1 - (rss / tss)
    rmse = RMSE(rss, y_pred.size)
    return r_square, rmse
```

- $R^2 = 1$ → perfect fit
- $R^2 = 0$ → the model does no better than always predicting the mean
- $R^2 < 0$ → the model is actively worse than the baseline

---

## 6. Visualising the Results

Numbers are useful, but plots make things click.

### The Regression Line

After training, we can overlay the predicted line on top of the actual data points. A well-fitted line should cut through the middle of the scatter:

```python
def plot_results(ax, X, y_pred, y_train):
    ax.plot(X, y_pred, label="prediction")
    ax.scatter(X, y_train, c="r", label="actual")
    ax.set_xlabel("mileage")
    ax.set_ylabel("price")
```

![Regression Line](/assets/img/posts/linear-regression/regression_line.png)
_The fitted regression line (blue) over the actual data points (red). The closer the points cluster around the line, the better the fit._

### The Convergence Curve

Logging MSE at every epoch lets us visualise the training process. A healthy convergence curve drops steeply at first, then flattens out as the model approaches the minimum:

```python
def plot_errors_convergence(ax, mse):
    ax.plot(np.arange(0, 1000, 1), mse)
    ax.set_xlabel("epochs")
    ax.set_ylabel("mean squared error")
```

![Convergence Curve](/assets/img/posts/linear-regression/convergence.png)
_MSE over 1000 epochs. The rapid drop followed by a plateau is the signature of a well-tuned gradient descent._

If the curve never flattens, you need more iterations. If it oscillates wildly, your learning rate is too high.

---

## 7. Keeping Training and Prediction Separate

Once training is done, the model is just two numbers: $\theta_0$ and $\theta_1$. There is no reason to retrain every time you want a prediction.

I saved them to a JSON file right after training:

```python
def write_values(theta0, theta1):
    s = {"theta0": theta0, "theta1": theta1}
    with open("params.json", "w") as f:
        json.dump(s, f)
```

Then in `predict.py`, I load them back and score new data without touching the training code at all:

```python
def get_params():
    with open("params.json") as f:
        return json.loads(f.read())
```

This separation matters. Training is expensive and happens once. Prediction is lightweight and happens on demand. Keeping them apart mirrors how real ML systems are structured train offline, serve online.

---

## 8. Takeaways

Building a model without any ML library forces you to understand every step. You can't just call `.fit()` and move on you have to know why normalisation matters, what gradient descent is actually doing, and what your evaluation metrics are really measuring.

The gap between reading about an algorithm and implementing it is where the real learning happens. Theory gives you the formula. Code makes you confront every assumption hidden inside it.

A few things that stuck with me:

- **Normalisation is not optional** it's what makes the algorithm stable, not a nice-to-have
- **The learning rate is more art than science** you feel it out through trial and error
- **R² puts everything in context** a low RMSE means nothing if the baseline is just as good
- **Separating train and predict early** makes your code cleaner and your thinking clearer
