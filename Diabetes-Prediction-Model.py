

"""Diabetes is a chronic medical condition that affects millions of people worldwide. 
Early detection and accurate prediction of diabetes can help in early intervention and 
better disease control. In this project, we will implement a Logistic Regression model 
from scratch to predict whether an individual has diabetes or not based on medical attributes.



We will use the Pima Indians Diabetes Dataset from the UCI Machine Learning Repository. 
The data contains several biometric factors such as blood glucose, BMI, and insulin, 
which are used as predictors of diabetes. We need to develop a classification model 
that predicts whether a patient has diabetes (1) or not (0).



Since the primary objective is to understand the mathematical and algorithmic concepts 
of Logistic Regression, we will implement the model without using machine learning 
libraries such as Scikit-Learn. We will use basic Python libraries such as NumPy and 
Pandas for numerical computations and data manipulation.

Logistic Regression is a supervised learning algorithm for binary classification problems. 
As opposed to Linear Regression, which makes predictions of continuous values, Logistic 
Regression predicts probabilities using the sigmoid function to guarantee outputs between 0 and 1. 
The hypothesis function for the model is  h(X)=σ(WᵀX+b), where W represents weights, X is the input, 
and b is the bias. As Mean Squared Error (MSE) for classification is non-convex, 
the cost function log-loss (binary cross-entropy) is used so that the model reduces the 
classification errors to be as minimal as possible. Optimization is achieved using the 
Gradient Descent that iteratively updates the weights from the gradients computed.
 Decision boundary is set using a threshold (most commonly 0.5), and the inputs are 
 labeled as 0 or 1. Logistic Regression has numerous applications in spam filtering, 
 medical diagnosis, customer churn prediction, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from google.colab import files

uploaded = files.upload()

df = pd.read_csv("DiabetesData.csv")
#Read the data file into data frame

#Handle missing value, replacing zeros in the data with the mean of the column
columns_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_to_impute:
    df[column] = df[column].replace(0, df[column].mean())

"""In the data preprocessing steps, first we handle the values 
which should not be zero in some attributes, but the dataset 
might have 0 in the place of missing values. In this step, we 
replace the zeros with the mean of the column as a way to handl
 missing data while ensuring the data intergrity."""

#Normalize the data
def normalize_column(column):
    return (column - column.mean()) / column.std()

df.iloc[:, :-1] = df.iloc[:, :-1].apply(normalize_column)

"""Then we proceed to standardize the feature value in the
 dataset because some features might vary significantly 
 from others. Performing Standarization sets the mean among 
 all features to 0, and standard deviation to 1. 
 Standardization makes the gradient decent more stable and 
 enhances model performance."""

#Get all the features except the last column
X = df.iloc[:, :-1].values
#Get the last column as target
Y = df.iloc[:, -1].values

"""The dataset contains the independent and dependent variable, 
we set X to store the independent variables and Y to store the 
dependent variables."""

#Calculating the split size
train_size = int(0.8 * len(X))
#Split features
X_train, X_test = X[:train_size], X[train_size:]
#Split target
Y_train, Y_test = Y[:train_size], Y[train_size:]

"""Splitting the data into 80/20. 80 percent of the data is 
used for training and 20% of the data is used for testing."""

#Defining logistical Regression from scratch
class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        #Setting learning rate
        self.learning_rate = learning_rate
        #Setting number of iterations
        self.iterations = iterations
        #Defining a variable for weight
        self.W = None
        #Defining a variable for bias
        self.b = None
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, A, Y):
        return -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    def fit(self, X, Y):
        #Getting the number of rows and columns
        self.m, self.n = X.shape
        #Initializing the weights to zero
        self.W = np.zeros(self.n)
        self.b = 0

        #Training using gradient descent
        for _ in range(self.iterations):
            #Linear combination and computing predictions
            A = self.sigmoid(np.dot(X, self.W) + self.b)
            #Calculate the error
            error = A - Y.T
            error = np.reshape(error, self.m)

            #Compute gradient updates
            dW = np.dot(X.T, error) / self.m
            db = np.sum(error) / self.m

            #Update weights and bias
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            #Compute and store loss
            self.loss_history.append(self.compute_loss(A, Y))

    def predict(self, X):
        Z = self.sigmoid(np.dot(X, self.W) + self.b)
        #Convert probability to binary value
        return np.where(Z > 0.5, 1, 0)

"""1. Constructor: The constructor (__init__ method) sets the learning rate 
as well as the number of iterations and the model parameters. The learning 
rate (learning_rate) is the step size in the gradient descent and iterations 
is the number of times the model parameters are updated. The weights (W) and 
the bias (b) are set to None as initial values and they will be determined 
during training. Furthermore, a list called loss_history is used to keep 
track of the loss at each iteration which can be useful in checking whether 
the model is converging.

2. Sigmoid function: The sigmoid function is used to covert linear input into a 
probability between 0 and 1.

3. Compute loss: The function compute_loss computes the binary cross-entropy 
loss (also called log-loss), which quantifies how well the model's predicted
 probabilities correspond to the true labels. This loss function punishes 
 incorrect predictions more when they are further away from the true label
   (0 or 1) to push the model to learn to make more confident and correct 
   predictions. The lower the loss, the better the model is performing.

4. fit function: The fit method is used to train the model where it updates 
the weights and bias iteratively through gradient descent. The number of 
training examples and features are determined. Intially, The the weights are 
set to zero and the bias is also set to zero. For each iteration, the model 
computes the linear transformation and uses the sigmoid function to come up 
with the predicted probabilities. Then we get the error by calculating the 
difference between the predicted values and the actual values. After computing 
the gradients, we use them to adjust the model's parameters. Repeating the 
process for specified number of iterations. The loss is recorded at each step 
to visualize and assess convergence.

5. Predict function: predict function utilizes the trained weights and biases 
to predict new data and applies the sigmoid function to determine probabilities 
and assign class labels. If the output is greater than 0.5, then it is classified 
as positive class else it is negative.
"""

#Training the custom model
custom_model = SimpleLogisticRegression(learning_rate=0.01, iterations=1000)
custom_model.fit(X_train, Y_train)
Y_pred_custom = custom_model.predict(X_test)

sklearn_model = LogisticRegression()
sklearn_model.fit(X_train, Y_train)
Y_pred_sklearn = sklearn_model.predict(X_test)

#Evaluating the custom model
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}\n")
    return accuracy

custom_accuracy = evaluate_model(Y_test, Y_pred_custom, "Custom Logistic Regression")
sklearn_accuracy = evaluate_model(Y_test, Y_pred_sklearn, "Sklearn Logistic Regression")

plt.plot(range(custom_model.iterations), custom_model.loss_history, label='Loss over iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve of Custom Logistic Regression')
plt.legend()
plt.show()

conf_matrix = np.zeros((2, 2))
conf_matrix[0, 0] = np.sum((Y_pred_custom == 0) & (Y_test == 0))
conf_matrix[0, 1] = np.sum((Y_pred_custom == 1) & (Y_test == 0))
conf_matrix[1, 0] = np.sum((Y_pred_custom == 0) & (Y_test == 1))
conf_matrix[1, 1] = np.sum((Y_pred_custom == 1) & (Y_test == 1))
print("Confusion Matrix for Custom Model:\n", conf_matrix)

fpr = []
tpr = []
thresh = np.linspace(0, 1, 100)
for threshold in thresh:
    Y_pred_prob = custom_model.sigmoid(np.dot(X_test, custom_model.W) + custom_model.b)
    Y_pred = np.where(Y_pred_prob > threshold, 1, 0)
    fp = np.sum((Y_pred == 1) & (Y_test == 0))
    tp = np.sum((Y_pred == 1) & (Y_test == 1))
    fn = np.sum((Y_pred == 0) & (Y_test == 1))
    tn = np.sum((Y_pred == 0) & (Y_test == 0))
    fpr.append(fp / (fp + tn))
    tpr.append(tp / (tp + fn))

roc_auc = np.trapz(tpr, fpr)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Custom Logistic Regression')
plt.legend()
plt.show()

# User input for prediction
print("Enter patient details:")
user_input = []
columns = df.columns[:-1]
for col in columns:
    value = float(input(f"{col}: "))
    normalized_value = (value - df[col].mean()) / df[col].std()
    user_input.append(normalized_value)

user_input = np.array(user_input).reshape(1, -1)
prediction = custom_model.predict(user_input)
print("Prediction: Diabetic" if prediction == 1 else "Prediction: Not Diabetic")

"""Insights and Challenges

1. The custom implementation of logistic regression in this assignment 
performed almost identical on all the metrics above. This also suggests 
that the implemented model is well-defined and straightforward algorithm. 
It also shows the practicality of this model, that this model can not only 
be used for educational purposes but also for real-world problems and scenarios.

2. We split the data manually in the above program which ensures more 
transparency. It also give us a better understanding of how the data 
is split between training and testing and ensures that no unintended 
biases enter while splitting the data since we are not using an external library.

3. Logistic regression assumes that all features of the dataset 
contribute equally to the prediction but some features might have 
larger values compared to others which might skew the prediction and 
give more weight to selective features. Normalizing the data as above 
was essential so that each feature contributes effectively and the 
algorithm works properly.
"""