import numpy as np
from sklearn.metrics import log_loss
# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# logistic loss function
def logistic_loss(w, X, y):
    N = X.shape[0]
    z = X @ w
    loss = np.mean(-y * z + np.log(1 + np.exp(z)))
    #print(loss)
    return loss

# compute gradient
def compute_gradient(w, X, y):
    N = X.shape[0]
    prob = sigmoid(X @ w)
    return X.T @ (prob - y) / N

def coordinate_descent(X, y, num_iterations=1000, learning_rate=0.01, tol=1e-1, verbose=True):
    """
    param:
        X: character matrix
        y: label vector (0/1)
        num_iterations: maximum number of iterations
        learning_rate: learning rate
        tol: convergence threshold (maximum absolute gradient)
        verbose: whether to print progress
    """
    # 初始化参数
    w = np.random.rand(X.shape[1])
    prev_loss = np.inf
    grad = compute_gradient(w, X, y)
    losses = []
    for iter in range(num_iterations):
        # choose the coordinate with the largest gradient
        j = np.argmax(np.abs(grad))
        
        # update the selected coordinate
        w[j] -= learning_rate * grad[j]
        
        # compute the new gradient
        grad = compute_gradient(w, X, y)
        
        # compute the current loss
        current_loss = logistic_loss(w, X, y)
        losses.append(current_loss)
        # check the convergence condition
        if np.max(np.abs(grad)) < tol:
            if verbose:
                print(f"After {iter+1} iterations, the method converage")
            break
        
        # print the progress
        if verbose and (iter+1) % 100 == 0:
            print(f"Iter {iter+1}: Loss = {current_loss:.4f}, Max|Grad| = {np.max(np.abs(grad)):.4f}")
    
    return w, losses
def random_feature_coordinate_descent(X, y, num_iterations=1000, learning_rate=0.01, tol=1e-1, verbose=True):
    """

    param:
        X: character matrix
        y: label vector (0/1)
        num_iterations: maximum number of iterations
        learning_rate: learning rate
        tol: convergence threshold (maximum absolute gradient)
        verbose: whether to print progress
    """
    # initialize the parameters
    w = np.random.rand(X.shape[1])
    prev_loss = np.inf
    grad = compute_gradient(w, X, y)
    losses = []
    for iter in range(num_iterations):
        # choose the coordinate with the largest gradient
        j = np.random.randint(0, X.shape[1])
        
        # update the selected coordinate
        w[j] -= learning_rate * grad[j]
        
        # compute the new gradient
        grad = compute_gradient(w, X, y)
        
        # compute the current loss
        current_loss = logistic_loss(w, X, y)
        losses.append(current_loss)
        # check the convergence condition
        if np.max(np.abs(grad)) < tol:
            if verbose:
                print(f"After {iter+1} iterations, the method converage")
            break
        
        # print the progress
        if verbose and (iter+1) % 100 == 0:
            print(f"Iter {iter+1}: Loss = {current_loss:.4f}, Max|Grad| = {np.max(np.abs(grad)):.4f}")
    
    return w, losses

# load data (please replace with the actual file path)
data = np.loadtxt('ucsd_hw/251A_hw/assignment2/data/wine_binary.data', delimiter=',')

# separate the features and labels
X = data[:, 1:]  # features
y = data[:, 0]   # labels
y = y-1
# add the intercept term and normalize the features
X = np.hstack([np.ones((X.shape[0], 1)), X])
mean = np.mean(X[:, 1:], axis=0)
std = np.std(X[:, 1:], axis=0)
X[:, 1:] = (X[:, 1:] - mean) / std

# run the coordinate descent
w, coord_losses = coordinate_descent(X, y, 
                      num_iterations=1000,
                      learning_rate=0.1,  
                      tol=1e-4)



_, random_coord_losses = random_feature_coordinate_descent(X, y)

# compute the prediction accuracy
pred_prob = sigmoid(X @ w)
pred = (pred_prob >= 0.5).astype(int)
accuracy = np.mean(pred == y)
print(f"Training accuracy: {accuracy*100:.2f}%")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty=None, solver='lbfgs')
model.fit(X, y)
y_pred_prob = model.predict_proba(X)[:, 1]
L_star = log_loss(y, y_pred_prob)
print(f"Final loss L*: {L_star}")


plt.figure(figsize=(10, 6))
plt.plot(coord_losses, label='Coordinate Descent')
plt.plot(random_coord_losses, label='Random-Feature Coordinate Descent')
plt.axhline(y=L_star, color='r', linestyle='--', label='L* (Logistic Regression)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration')
plt.legend()

plt.savefig('/root/code/ucsd_hw/251A_hw/assignment2/figure/loss_curve_1.png')
plt.show()