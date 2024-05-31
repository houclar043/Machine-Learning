from matplotlib import pyplot as plt
from utils import *
from scipy.linalg import sqrtm

import numpy as np

from starter_code.utils import load_public_test_csv, load_train_csv, \
    load_train_sparse, load_valid_csv


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    for i in range(len(s)):
        print(s[i][i])
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]
    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    for j in range(len(u[n])):
        u[n][j] -= lr * - z[q][j]*(c - np.dot(u[n], z[q]))
    for j in range(len(z[q])):
        z[q][j] -= lr * - u[n][j]*(c- np.dot(u[n], z[q]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, valid_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    train_loss = []
    val_loss = []
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        if i % 2000 == 0:
            train_loss.append(squared_error_loss(train_data, u, z))
            val_loss.append(squared_error_loss(valid_data, u, z))
    mat = np.matmul(u, np.transpose(z))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, train_loss, val_loss


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    total = 0
    num_correct = 0
    mats = []
    mats.append(svd_reconstruct(train_matrix, 1))
    mats.append(svd_reconstruct(train_matrix, 7))
    mats.append(svd_reconstruct(train_matrix, 15))
    mats.append(svd_reconstruct(train_matrix, 30))
    mats.append(svd_reconstruct(train_matrix, 50))
    ks = [1, 7, 15, 30, 50]
    for i in range(5):
        total = 0
        num_correct = 0
        for j in range(len(val_data["question_id"])):
            total +=1
            pred = mats[i][val_data["user_id"][j]][val_data["question_id"][j]]
            target = val_data["is_correct"][j]
            if (pred >= 0.5 and target == 1) or (pred < 0.5 and target == 0):
                num_correct += 1
        print("Validation accuracy for k = " + str(ks[i]) + " is " + str(num_correct/total))
    test_mat = svd_reconstruct(train_matrix, 7)
    total = 0
    num_correct = 0
    for j in range(len(test_data["question_id"])):
        total +=1
        if abs(test_mat[test_data["user_id"][j]][test_data["question_id"][j]]- test_data["is_correct"][j]) <= 0.5:
            num_correct +=1
    print("Test accuracy for k* = 7 is " + str(num_correct/total))



    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    mats = []
    mats.append(als(train_data, val_data, 1, .05, 100000)[0])
    mats.append(als(train_data, val_data, 15, .05, 100000)[0])
    mats.append(als(train_data, val_data, 30, .05, 100000)[0])
    mats.append(als(train_data, val_data, 50, .05, 100000)[0])
    mat, train_loss, val_loss = als(train_data, val_data, 80, 0.05, 100000)
    mats.append(mat)
    ks = [1, 15, 30, 50, 80]
    for i in range(5):
        total = 0
        num_correct = 0
        for j in range(len(val_data["question_id"])):
            total +=1
            if abs(mats[i][val_data["user_id"][j]][val_data["question_id"][j]] - val_data["is_correct"][j]) <= 0.5:
                num_correct += 1
        print("Validation accuracy for k = " + str(ks[i]) + " is " + str(num_correct/total))
    test_mat = mats[3]
    total = 0
    num_correct = 0
    for j in range(len(test_data["question_id"])):
        total +=1
        if abs(test_mat[test_data["user_id"][j]][test_data["question_id"][j]]- test_data["is_correct"][j]) <= 0.5:
            num_correct +=1
    print("Test accuracy for k* = 80 is " + str(num_correct/total))
    plt.plot(train_loss, label = "Train Loss")
    plt.plot(val_loss, label = "Validation Loss")
    plt.legend()
    plt.xlabel("Iteration number times times 2000")
    plt.ylabel("Squared loss")

    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
