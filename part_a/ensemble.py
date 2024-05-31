# TODO: complete this file.
import random
import item_response
import matrix_factorization
import knn

from starter_code.utils import load_public_test_csv, load_train_csv, \
    load_train_sparse, load_valid_csv

train_matrix = load_train_sparse("../data").toarray()
train_data = load_train_csv("../data")
val_data = load_valid_csv("../data")
test_data = load_public_test_csv("../data")

def bootstrap(train_data):
    num_samples = len(train_data["user_id"])
    data_one = {"user_id": [], "question_id": [], "is_correct": []}
    data_two = {"user_id": [], "question_id": [], "is_correct": []}
    data_three = {"user_id": [], "question_id": [], "is_correct": []}
    for i in range(num_samples):
        rand_one = random.randint(0, num_samples-1)
        rand_two = random.randint(0, num_samples-1)
        rand_three = random.randint(0, num_samples-1)
        data_one["user_id"].append(train_data["user_id"][rand_one])
        data_one["question_id"].append(train_data["question_id"][rand_one])
        data_one["is_correct"].append(train_data["is_correct"][rand_one])
        data_two["user_id"].append(train_data["user_id"][rand_two])
        data_two["question_id"].append(train_data["question_id"][rand_two])
        data_two["is_correct"].append(train_data["is_correct"][rand_two])
        data_three["user_id"].append(train_data["user_id"][rand_three])
        data_three["question_id"].append(train_data["question_id"][rand_three])
        data_three["is_correct"].append(train_data["is_correct"][rand_three])
    return data_one, data_two, data_three

def make_predictions(theta, beta, data):
    preds = []
    for i in range(len(data["user_id"])):
        if theta[data["user_id"][i]]-beta[data["question_id"][i]] >= 0:
            preds.append(1)
        else:
            preds.append(0)
    return preds

def average_prediction(pred_one, pred_two, pred_three):
    preds = []
    for i in range(len(pred_one)):
        if pred_one[i] + pred_two[i] + pred_three[i] >= 2:
            preds.append(1)
        else:
            preds.append(0)
    return preds

def get_accuaracy(data, predictions):
    total_correct = 0
    total = 0
    for i in range(len(predictions)):
        if predictions[i] == data["is_correct"][i]:
            total_correct +=1
        total += 1
    return total_correct / total


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    data_one, data_two, data_three = bootstrap(train_data)
    theta_one, beta_one, val_acc = item_response.irt(data_one, val_data, 0.01, 20)
    theta_two, beta_two, val_acc = item_response.irt(data_two, val_data, 0.01, 20)
    theta_three, beta_three, val_acc = item_response.irt(data_three, val_data, 0.01, 20)
    val_pred_one = make_predictions(theta_one, beta_one, val_data)
    val_pred_two = make_predictions(theta_two, beta_two, val_data)
    val_pred_three = make_predictions(theta_three, beta_three, val_data)
    val_pred = average_prediction(val_pred_one, val_pred_two, val_pred_three)
    val_acc = get_accuaracy(val_data, val_pred)
    test_pred_one = make_predictions(theta_one, beta_one, test_data)
    test_pred_two = make_predictions(theta_two, beta_two, test_data)
    test_pred_three = make_predictions(theta_three, beta_three, test_data)
    test_pred = average_prediction(test_pred_one, test_pred_two, test_pred_three)
    test_acc = get_accuaracy(test_data, test_pred)
    print("Validation accuracy is " + str(val_acc))
    print("Test accuracy is " + str(test_acc))


if __name__ == "__main__":
    main()
