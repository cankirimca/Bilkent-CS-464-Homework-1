import numpy as np
import pandas as pd
import math
import sys
import time

#takes parameter from command line
#size_of_vocabulary = int(sys.argv[1])

#fixed vocabulary size
size_of_vocabulary = 3458

spam_count = 0
ham_count = 0
words_in_spam = np.zeros(size_of_vocabulary, dtype=int)
words_in_ham = np.zeros(size_of_vocabulary, dtype=int)


def train(train_features, train_labels):
    global spam_count
    global ham_count
    for index1, train_sample in enumerate(train_features.itertuples(index=False, name='Pandas')):
        sample_label = int(train_labels.iloc[index1])  # indicates whether the message is spam
        if sample_label == 0:
            ham_count += 1
            for index2, feature in enumerate(train_sample):
                if feature > 0:
                    words_in_ham[index2] += int(feature)
        else:
            spam_count += 1
            for index2, feature in enumerate(train_sample):
                if feature > 0:
                    words_in_spam[index2] += int(feature)
                    # for entry in words_in_ham:
        # print(entry)
    # print("hams:", ham_count, ", spams:", spam_count)


def predict(test_features):
    ham_ratio = ham_count / (ham_count + spam_count)
    spam_ratio = spam_count / (ham_count + spam_count)
    predictions = np.zeros(test_features.size, dtype=int)

    for index1, test_sample in enumerate(test_features.itertuples(index=False, name='Pandas')):
        spam_likelihood = np.log(spam_ratio)
        ham_likelihood = np.log(ham_ratio)
        for index2, feature in enumerate(test_sample):
            if feature > 0:
                if words_in_spam[index2] == 0:
                    spam_likelihood += float('-inf')
                else:
                    spam_likelihood += (feature) * np.log((words_in_spam[index2]) / (np.sum(words_in_spam)))
                if words_in_ham[index2] == 0:
                    ham_likelihood += float('-inf')
                else:
                    ham_likelihood += (feature) * np.log((words_in_ham[index2]) / (np.sum(words_in_ham)))

        # print("spam_likelihood:", spam_likelihood)
        # print("ham_likelihood:", ham_likelihood, "\n")
        if spam_likelihood > ham_likelihood:
            predictions[index1] = 1
    return predictions

begin = time.time()
train_features = pd.read_csv("sms_train_features.csv")
train_labels = pd.read_csv("sms_train_labels.csv")
test_features = pd.read_csv("sms_test_features.csv")
test_labels = pd.read_csv("sms_test_labels.csv")

train_features.drop(train_features.columns[0], axis=1, inplace=True)
test_features.drop(test_features.columns[0], axis=1, inplace=True)
train_labels.drop(train_labels.columns[0], axis=1, inplace=True)
test_labels.drop(test_labels .columns[0], axis=1, inplace=True)



train_time_start = time.time()
train(train_features, train_labels)
train_time_end = time.time()

pred_time_start = time.time()
predictions = predict(test_features)
pred_time_end = time.time()

number_of_guesses = 0
number_of_correct_guesses = 0
print("hams:", ham_count, ", spams:", spam_count)
#print("predictions:")
fp = 0
tp = 0
fn = 0
tn = 0

for index, item in enumerate(test_labels.itertuples(index=False, name='Pandas')):
    prediction = predictions[index]
    if prediction == 1:
      if prediction == item:
        tp += 1
        number_of_correct_guesses += 1
      else:
        fp += 1

    else:
      if prediction == item:
        tn += 1
        number_of_correct_guesses += 1
      else:
        fn += 1
    number_of_guesses += 1
accuracy = number_of_correct_guesses/number_of_guesses
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("Accuracy: ",accuracy)
print("Training time:", train_time_end - train_time_start)
print("Prediction time:", pred_time_end - pred_time_start)
end = time.time()
print("Total time of execution:", end - begin)