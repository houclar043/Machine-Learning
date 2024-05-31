import math

from matplotlib import pyplot as plt
from utils import *

import numpy as np

from starter_code.utils import load_public_test_csv, load_train_csv, \
    load_train_sparse, load_valid_csv


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i in (range(len(data["user_id"]))):
        log_lklihood += data["is_correct"][i] * np.log(sigmoid(
            theta[data["user_id"][i]] - beta[data["question_id"][i]])) + (
                                    1 - data["is_correct"][i]) * np.log(
            1 - sigmoid(
                theta[data["user_id"][i]] - beta[data["question_id"][i]]))
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    iterations = 100
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    theta_grad = np.zeros(len(theta))
    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        is_correct = data["is_correct"][i]
        theta_grad[user_id] -= is_correct - sigmoid(theta[user_id] - beta[question_id])
    for i in range(len(theta)):
        theta[i] -= lr * theta_grad[i]

    beta_grad = np.zeros(len(beta))
    for i in range(len(data["user_id"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        is_correct = data["is_correct"][i]
        beta_grad[question_id] += is_correct - sigmoid(theta[user_id] - beta[question_id])
    for i in range(len(beta)):
        beta[i] -= lr * beta_grad[i]

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.rand(542)
    beta = np.random.rand(1774)

    val_acc_lst = []
    train_Nllk = []
    val_Nllk = []
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)
        train_Nllk.append(neg_lld)
        val_Nllk.append(neg_log_likelihood(val_data, theta, beta))

    plt.plot(train_Nllk)
    plt.xlabel("Iteration number")
    plt.ylabel("Negative log likelihood")
    plt.title("Training negative log likelihood as a function of iteration")
    plt.show()
    plt.plot(val_Nllk)
    plt.xlabel("Iteration number")
    plt.ylabel("Negative log likelihood")
    plt.title("Validation negative log likelihood as a function of iteration")
    plt.show()

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    # print(theta[0])
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    theta, beta, val_acc_list = irt(train_data, val_data, 0.01, 35)
    print("Final validation accuracy is " + str(evaluate(val_data, theta, beta)))
    print("Final test accuracy is " + str(evaluate(test_data, theta, beta)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)#
    j1 = []
    j2 = []
    j3 = []
    for i in range(len(theta)):
        j1.append(sigmoid(theta[i]-beta[1410]))
        j2.append(sigmoid(theta[i]-beta[1444]))
        j3.append(sigmoid(theta[i]-beta[781]))
    plt.plot(j1, label= "Question 1410")
    plt.plot(j2, label = "Question 1444")
    plt.plot(j3, label = "Question 781")
    plt.legend()
    plt.xlabel("User Number")
    plt.ylabel("Predicted chance of success")
    plt.show()
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
