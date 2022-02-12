import math
import pandas as pd
import time
total_train_time = 0
total_validation_time = 0


def knn_classifier(train_features, train_labels, k, test_sample):
    global total_train_time
    global total_validation_time

    #the following code block was added for deriving a training time to mention in the report
    train_time_begin = time.time()
    fitted_train_features = train_features
    fitted_train_labels = train_labels
    fitted_test_sample = test_sample
    train_time_end = time.time()

    total_train_time += train_time_end - train_time_begin

    distances = []
    point2 = []

    val_time_begin = time.time()

    for feature in fitted_test_sample:
        point2.append(float(feature))

    for fitted_train_sample in fitted_train_features.itertuples(index=True, name='Pandas'):
        point1 = []
        for feature in fitted_train_sample[1:]:
            point1.append(float(feature))
        distances.append(distance(point1, point2))

    numbered_dist = list(enumerate(distances))
    sorted_dist = sorted(numbered_dist, key=lambda i: i[1])

    zero_count = 0
    one_count = 0
    for item in sorted_dist[:k]:
        if int(fitted_train_labels.iloc[item[0]]) == 0:
            zero_count += 1
        elif int(fitted_train_labels.iloc[item[0]]) == 1:
            one_count += 1
    val_time_end = time.time()
    total_validation_time += val_time_end - val_time_begin
    return 0 if zero_count > one_count else 1


# sample consists of many features, treated like a point with many dimensions
def distance(sample1, sample2):
    sum_dist = 0
    for i in range(len(sample1)):
        sum_dist += math.pow(sample1[i] - sample2[i], 2)
    return math.sqrt(sum_dist)


begin = time.time()
train_features = pd.read_csv("diabetes_train_features.csv")
train_labels = pd.read_csv("diabetes_train_labels.csv")
test_features = pd.read_csv("diabetes_test_features.csv")
test_labels = pd.read_csv("diabetes_test_labels.csv")

train_features.drop(train_features.columns[0], axis=1, inplace=True)
test_features.drop(test_features.columns[0], axis=1, inplace=True)
train_labels.drop(train_labels.columns[0], axis=1, inplace=True)
test_labels.drop(test_labels.columns[0], axis=1, inplace=True)

correct_guess_count = 0
guess_count = 0


accuracy = 0
fp = 0
tp = 0
fn = 0
tn = 0
for i in range(0, len(test_features)):

    guess = knn_classifier(train_features, train_labels, 9, test_features.iloc[i])
    real_value = test_labels['Outcome'][i]
    if int(guess) == 1:
        if int(real_value) == 1:
            tp += 1
            correct_guess_count += 1
        else:
            fp += 1
    else:
        if int(real_value) == 0:
            tn += 1
            correct_guess_count += 1
        else:
            fn += 1
    guess_count += 1
accuracy = correct_guess_count / guess_count
print("With all features:")
print("Correct guesses:", correct_guess_count)
print("All guesses:", guess_count)
print("Accuracy:", accuracy)
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("Training time:", total_train_time)
print("Validation time:", total_validation_time)
total_train_time = 0
total_validation_time = 0

max_accuracy = accuracy
step = 1
while True:
    max_accuracy_feature_index = -1
    for k in range(len(train_features.columns)):
        fp = 0
        tp = 0
        fn = 0
        tn = 0
        guess_count = 0
        correct_guess_count = 0
        new_train_features = train_features.drop(train_features.columns[k], axis=1, inplace=False)
        new_test_features = test_features.drop(test_features.columns[k], axis=1, inplace=False)
        for i in range(0, len(test_features)):
            guess = knn_classifier(new_train_features, train_labels, 9, new_test_features.iloc[i])
            real_value = test_labels['Outcome'][i]
            if int(guess) == 1:
                if int(real_value) == 1:
                    tp += 1
                    correct_guess_count += 1
                else:
                    fp += 1
            else:
                if int(real_value) == 0:
                    tn += 1
                    correct_guess_count += 1
                else:
                    fn += 1
            guess_count += 1
        new_accuracy = correct_guess_count / guess_count
        if new_accuracy > max_accuracy:
            max_accuracy_feature_index = k
            max_accuracy = new_accuracy
            print("-------------------------------------------------")
            print("Step", step)
            print("Feature", train_features.columns[max_accuracy_feature_index], "is eliminated")
            print("Correct guesses:", correct_guess_count)
            print("All guesses:", guess_count)
            print("Accuracy:", max_accuracy)
            print("True Positives:", tp)
            print("True Negatives:", tn)
            print("False Positives:", fp)
            print("False Negatives:", fn)
            print("Training time:", total_train_time)
            print("Validation time:", total_validation_time)
            step += 1
            total_train_time = 0
            total_validation_time = 0
        else:
            total_train_time = 0
            total_validation_time = 0

    if max_accuracy_feature_index != -1:
        train_features.drop(train_features.columns[max_accuracy_feature_index], axis=1, inplace=True)
        test_features.drop(test_features.columns[max_accuracy_feature_index], axis=1, inplace=True)
        accuracy = max_accuracy
    else:
        break

end = time.time()
print("time:", end - begin)


