
# Sagi Eden 204278931
# Einat Shusterman 307848952

import numpy as np
import sys
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(h, y):
    return abs((h-y)).mean()


def gradient(X, y, theta):
    """
    :param X: matrix X
    :param y: vector y
    :param theta: vector of parameters θ
    :return: vector of partial derivatives
    """
    scores = np.dot(X, theta)
    predictions = sigmoid(scores)
    output_error = y - predictions
    gradient = np.dot(X.T, output_error) / X.shape[0]
    return gradient


def gradient_descent(X, y, alpha=0.1, max_iterations=1000, threshold=1e-5):
    """

    :param X: matrix X
    :param y: vector y
    :param alpha: step size alpha
    :param max_iterations: maximal number of iterations
    :param threshold: delta threshold
    :return: The function first adds a dummy variable of ‘1’ as the first column in X to take care of
            the bias. Then, initialize the parameter θ to vector of zeros of the (new) length of
            features. In each iteration, update θ by subtracting α times the gradient. If the
            maximal number of iterations was reached, or the change in θ is settled, stop
            iterating. Change in θ is measured as ||𝜃#$% − 𝜃# ||( (root square of sum of squares of
            differences).The function returns the last θ.
    """
    X['bias'] = np.ones(X.shape[0], dtype=int)
    theta = np.zeros(X.shape[1])

    for i in range(max_iterations):
        grad = gradient(X, y, theta)
        updated_theta = theta - alpha * grad
        if np.linalg.norm(updated_theta-theta) < threshold:
            break
        theta = updated_theta
    return theta


def predict(X, theta):
    """
    :param X: matrix X
    :param theta: parameter θ
    :return: binary vector of length of number of rows in
            X. Don’t forget to add a dummy variable to X as the first column.
    """
    X['bias'] = np.ones(X.shape[0], dtype=int)
    z = np.dot(X, theta)
    h = sigmoid(z)
    predictions = list(map(lambda x: 1 if x > 0.5 else 0, h))
    return predictions


def calculate_model_accuracy(x, theta, y):
    y_test_predictions = predict(x, theta)
    accuracy = (y_test_predictions != y).sum() / len(y)
    return accuracy


def normalize_data_df(data_df):
    for column in data_df.iloc[:, :-1]:
        feat_mean = data_df[column].mean()
        feat_std = data_df[column].std()
        data_df[column] = (data_df[column] - feat_mean) / feat_std
    return data_df


def main():
    """
    :return: Main. Implement a main function, that will be called when file is executed. The
            function read a comma separated file text file (.csv) that was given as the only
            argument to the script. The file contains a header line, and the last column represent
            the class as 0 or 1. After loading the file, the function split it randomly to 80% rows
            as training, and the remaining 20% for testing.
            Then, the function train a logistic regression using the training set. Then, the
            function prints to the standard output the accuracy of the model estimated on the
            testing set. Accuracy is defined as the fraction of mislabeled elements.
    """

    np.seterr(over='ignore')
    if len(sys.argv) < 2:
        print("not given Dataset as input - stop script")
        return

    # file_input = 'ex1data3.csv'
    file_input = sys.argv[1]
    df = pd.read_csv(file_input)
    df = normalize_data_df(df)
    train = df.sample(frac=0.8, random_state=8)
    test = df.drop(train.index)

    x_train = train.iloc[:, 0:-1]
    x_test = test.iloc[:, 0:-1]
    y_train = train.iloc[:, -1]
    y_test = test.iloc[:, -1]

    theta = gradient_descent(x_train, y_train, alpha=0.1, max_iterations=1000, threshold=1e-5)
    # accuracy = calculate_model_accuracy(x_train, theta, y_train)
    accuracy = calculate_model_accuracy(x_test, theta, y_test)
    print("The accuracy of the model is: "+str(accuracy))


if __name__ == "__main__":
    main()
