import sys
from math import log
from random import randint, choice
from collections import Counter


class Node_Decisiont:
    def __init__(self, feat=None, tshld=None, left=None, right=None, value=None):
        self.feat = feat
        self.tshld = tshld
        self.left = left
        self.right = right
        self.value = value

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            row = list(map(float, line.strip().split()))
            data.append(row)
    feats = [row[:-1] for row in data]
    labels = [row[-1] for row in data]
    return feats, labels

def inf_gain(y, left_y, right_y):
    entropy_before = calculate_entropy(y)
    entropy_after = (len(left_y) / len(y)) * calculate_entropy(left_y) + (len(right_y) / len(y)) * calculate_entropy(right_y)
    return entropy_before - entropy_after

def calculate_entropy(y):
    cls_cnt = Counter(y)
    entropy = 0
    for count in cls_cnt.values():
        probability = count / len(y)
        entropy -= probability * log_base_2(probability)
    return entropy

def log_base_2(x):
    return 0 if x == 0 else log(x, 2)


def build_tree(feats, labels, option, depth=0, max_depth=5):
    if depth >= max_depth or len(set(labels)) == 1:
        return Node_Decisiont(value=labels[0])

    if option == 'optimized':
        best_feat, best_tshld = find_best_split(feats, labels)
    elif option == 'randomized':
        best_feat, best_tshld = find_random_split(feats, labels)
    else:
        raise ValueError("Invalid option")

    left_mask = [x[best_feat] <= best_tshld for x in feats]
    right_mask = [not x for x in left_mask]

    if all(not mask for mask in left_mask) or all(not mask for mask in right_mask):
        return Node_Decisiont(value=Counter(labels).most_common(1)[0][0])

    left_tree = build_tree([feats[i] for i in range(len(feats)) if left_mask[i]],
                           [labels[i] for i in range(len(labels)) if left_mask[i]],
                           option, depth + 1, max_depth)
    right_tree = build_tree([feats[i] for i in range(len(feats)) if right_mask[i]],
                            [labels[i] for i in range(len(labels)) if right_mask[i]],
                            option, depth + 1, max_depth)

    if left_tree.value is not None and len([True for mask in left_mask if mask]) < 50:
        return Node_Decisiont(value=Counter(labels).most_common(1)[0][0])

    if right_tree.value is not None and len([True for mask in right_mask if mask]) < 50:
        return Node_Decisiont(value=Counter(labels).most_common(1)[0][0])

    return Node_Decisiont(feat=best_feat, tshld=best_tshld, left=left_tree, right=right_tree)



def find_best_split(feats, labels):
    best_feat = None
    best_tshld = None
    max_info_gain = -float('inf')

    for feat in range(len(feats[0])):
        tshlds = sorted(set(x[feat] for x in feats))
        for tshld in tshlds:
            left_mask = [x[feat] <= tshld for x in feats]
            right_mask = [not x for x in left_mask]

            info_gain = inf_gain(labels, [labels[i] for i in range(len(labels)) if left_mask[i]],
                                          [labels[i] for i in range(len(labels)) if right_mask[i]])

            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feat = feat
                best_tshld = tshld

    return best_feat, best_tshld

def find_random_split(feats, labels):
    random_feat = randint(0, len(feats[0]) - 1)
    tshlds = sorted(set(x[random_feat] for x in feats))
    random_tshld = choice(tshlds)
    return random_feat, random_tshld

def predict(tree, x):
    if tree.value is not None:
        return tree.value

    if x[tree.feat] <= tree.tshld:
        return predict(tree.left, x)
    else:
        return predict(tree.right, x)

def predict_forest(forest, x):
    predictions = [predict(tree, x) for tree in forest]
    return Counter(predictions).most_common(1)[0][0]

def evlt(tree, feats, labels):
    correct = 0
    for i in range(len(feats)):
        prediction = predict(tree, feats[i])
        if prediction == labels[i]:
            correct += 1
    accy = correct / len(feats)
    return accy

def evlt_forest(forest, feats, labels):
    correct = 0
    for i in range(len(feats)):
        prediction = predict_forest(forest, feats[i])
        if prediction == labels[i]:
            correct += 1
    accy = correct / len(feats)
    return accy

def main(training_file, test_file, option):
    train_feats, train_labels = load_data(training_file)
    test_feats, test_labels = load_data(test_file)

    if option == 'optimized' or option == 'randomized':
        model = build_and_evaluate_tree(train_feats, train_labels, test_feats, test_labels, option)
        print_results_for_model(model, test_feats, test_labels)
    elif option == 'forest3' or option == 'forest15':
        num_trees = int(option.replace('forest', ''))
        model = build_and_evaluate_forest(train_feats, train_labels, test_feats, test_labels, num_trees)
        print_results_for_forest(model, test_feats, test_labels)
    else:
        raise ValueError("Invalid option")

    overall_accy = model['overall_accy']
    print(f"Classification accuracy = {overall_accy}")

def print_results(index, predicted_class, true_class, accy):
    print(f"Object Index = {index}, Result = {predicted_class}, True Class = {true_class}, accy = {accy}")


def print_results_for_model(model, feats, labels):
    for i in range(len(feats)):
        predicted_class = predict(model['model'], feats[i])
        accy = 1 if predicted_class == labels[i] else 0
        print_results(i, predicted_class, labels[i], accy)

def print_results_for_forest(model, feats, labels):
    for i in range(len(feats)):
        predicted_class = predict_forest(model['model'], feats[i])
        accy = 1 if predicted_class == labels[i] else 0
        print_results(i, predicted_class, labels[i], accy)

def build_and_evaluate_tree(train_feats, train_labels, test_feats, test_labels, option):
    tree = build_tree(train_feats, train_labels, option)
    overall_accy = evlt(tree, test_feats, test_labels)
    return {'model': tree, 'overall_accy': overall_accy}

def build_and_evaluate_forest(train_feats, train_labels, test_feats, test_labels, num_trees):
    forest = [build_tree(train_feats, train_labels, 'randomized') for _ in range(num_trees)]
    overall_accy = evlt_forest(forest, test_feats, test_labels)
    return {'model': forest, 'overall_accy': overall_accy}

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python decision_tree.py <training_file> <test_file> <option>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
