import numpy as np
from matplotlib import pyplot as plt
from numpy import argmax
from scipy.linalg import sqrtm

import item_response
from starter_code.utils import load_public_test_csv, load_train_csv, \
    load_train_sparse, \
    load_valid_csv


def main():
    """
    Does the thing
    :return:
    """
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    theta, beta, val_acc_list = item_response.irt(train_data, val_data, 0.01,
                                                  20)

    #Code for part 3
    #new_dict = {"user_id": [], "question_id": [], "is_correct": []}
    #for i in range(train_matrix.shape[0]):
    #    for j in range(train_matrix.shape[1]):
    #        new_dict["user_id"].append(i)
    #        new_dict["question_id"].append(j)
    #        if theta[i] - beta[j] >= 0:
    #            correct = 1
    #        else:
    #            correct = 0
    #        new_dict["is_correct"].append(correct)
    #theta, beta, val_acc_list = item_response.irt(new_dict, val_data, 0.01, 4)
    #print("Final validation accuracy is " + str(item_response.evaluate(val_data, theta, beta)))
    #print("Final test accuracy is " + str(item_response.evaluate(test_data, theta, beta)))
    val_acc = []
    test_acc = []
    for l in range(5):
        val_acc.append([])
        test_acc.append([])
        for i in range(train_matrix.shape[0]):
            for j in range(train_matrix.shape[1]):
                train_matrix[i][j] = item_response.sigmoid(theta[i] - beta[j]) - 0.5
                # train_matrix[i][j] = 0
        for i in range(len(train_data["is_correct"])):
            if (train_data["is_correct"][i]):
                train_matrix[train_data["user_id"][i]][
                    train_data["question_id"][i]] = 0.5*(l+1)
            else:
                train_matrix[train_data["user_id"][i]][
                    train_data["question_id"][i]] = -0.5*(l+1)
        ks = [1, 2, 3, 4, 5, 7, 9, 11]
        for k in ks:
            Q, s, Ut = np.linalg.svd(train_matrix)
            s = np.diag(s)
            # Choose top k eigenvalues.
            s = s[0:k, 0:k]
            Q = Q[:, 0:k]
            Ut = Ut[0:k, :]
            s_root = sqrtm(s)

            # Reconstruct the matrix.
            reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
            reconst_matrix = reconst_matrix + 0.5
            total = 0
            num_correct = 0
            for j in range(len(val_data["question_id"])):
                total += 1
                pred = reconst_matrix[val_data["user_id"][j]][
                    val_data["question_id"][j]]
                target = val_data["is_correct"][j]
                if (pred >= 0.5 and target == 1) or (pred < 0.5 and target == 0):
                    num_correct += 1
            print("Validation accuracy for k = " + str(k) + " and l = " + str(l +1) + " is " + str(
                num_correct / total))
            val_acc[l].append(num_correct/total)
            for j in range(len(test_data["question_id"])):
                total += 1
                pred = reconst_matrix[test_data["user_id"][j]][
                    test_data["question_id"][j]]
                target = test_data["is_correct"][j]
                if (pred >= 0.5 and target == 1) or (pred < 0.5 and target == 0):
                    num_correct += 1
            #print("Test accuracy for k = " + str(k) + " is " + str(num_correct/total))
            test_acc[l].append(num_correct/total)
    val_acc = np.asarray(val_acc)
    lstar, kstar = np.unravel_index(val_acc.argmax(), val_acc.shape)
    print("The chosen value of k* is " + str(ks[kstar]) + " and l* = " + str(lstar + 1) + " with validation accuracy " + str(val_acc[lstar][kstar]) + " and test accuracy " +
            str(test_acc[lstar][kstar]))
    fig = plt.figure()
    ax = fig.add_subplot(projection = "3d")
    X, Y = np.meshgrid(ks, range(1, 6))
    ax.plot_surface(X, Y, val_acc, cmap = 'viridis', alpha = 0.7)
    ax.set_xlabel("Value of K")
    ax.set_ylabel("Value of L")
    ax.set_zlabel("Validation accuracy")
    plt.show()


if __name__ == "__main__":
    main()
