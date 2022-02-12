import numpy as np
import pandas as pd
import time
import sys

#takes parameter from command line
#size_of_vocabulary = int(sys.argv[1])

#fixed vocabulary size
size_of_vocabulary = 3458

spam_count = 0
ham_count = 0
occurs_in_spam = np.zeros(size_of_vocabulary, dtype=int)
occurs_in_ham = np.zeros(size_of_vocabulary, dtype=int)

def count_hams_and_spams(train_features):
  global ham_count
  global spam_count
  for index1, train_sample in enumerate(train_features.itertuples(index=False, name='Pandas')):
    sample_label = int(train_labels.iloc[index1]) #indicates whether the message is spam
    if sample_label == 0:
      ham_count += 1
    else:
      spam_count += 1


def train(train_features, train_labels, used_feature_index, used_feature_count):
  global occurs_in_ham
  global occurs_in_spam
  occurs_in_spam = np.zeros(size_of_vocabulary, dtype=int)
  occurs_in_ham = np.zeros(size_of_vocabulary, dtype=int)

  np_index = np.asarray(range(used_feature_count))
  used_feature_index = np.asarray(used_feature_index)

  np_train = train_features.to_numpy()

  for index1, train_sample in enumerate(np_train):
    sample_label = int(train_labels.iloc[index1]) #indicates whether the message is spam
    #indexed_train_sample = enumerate(train_sample)
    if sample_label == 0:
        feature_index = used_feature_index[np_index, 0].astype(int)
        occurs_in_ham[feature_index] = occurs_in_ham[feature_index] + (train_sample[feature_index] > 0)
    else:
        feature_index = used_feature_index[np_index, 0].astype(int)
        occurs_in_spam[feature_index] = occurs_in_spam[feature_index] + (train_sample[feature_index] > 0)

#the double for loop implementation mentioned in the report
#to switch to this implementation, uncomment the following segment and comment out the for loop between lines 33-41
"""
  for index1, train_sample in enumerate(train_features.itertuples(index=False, name='Pandas')):
    sample_label = int(train_labels.iloc[index1])  # indicates whether the message is spam
    # indexed_train_sample = enumerate(train_sample)
    if sample_label == 0:
      for i in np_index:
        feature_index = used_feature_index[i][0].astype(int)
        if train_sample[feature_index] > 0:
          occurs_in_ham[feature_index] += 1
    else:
      for i in np_index:
        feature_index = used_feature_index[i][0].astype(int)
        if train_sample[feature_index] > 0:
          occurs_in_spam[feature_index] += 1
"""

def predict(test_features, used_feature_index, used_feature_count):
  ham_ratio = ham_count/(ham_count+spam_count)
  spam_ratio = spam_count/(ham_count+spam_count)
  predictions = np.zeros(test_features.size, dtype=int)

  np_test = test_features.to_numpy()
  for index1, test_sample in enumerate(np_test):
    spam_likelihood = 1
    ham_likelihood = 1
    #print(test_sample)
    indexed_test_sample = enumerate(test_sample)

    for i in range(used_feature_count):
      index2 = used_feature_index[i][0]
      feature = test_sample[index2]
      #print("ufi:", used_feature_index[i][0])
      #print("f: ", feature)
      if feature > 0:
        spam_likelihood *= ((feature) * ((occurs_in_spam[index2]) / (spam_count)))
        ham_likelihood *= ((feature) * ((occurs_in_ham[index2]) / (ham_count)))
      else:
        ham_likelihood *= ((1 - feature) * (1 - ((occurs_in_ham[index2]) / (ham_count))))
        spam_likelihood *= ((1 - feature) * (1 - ((occurs_in_spam[index2]) / (spam_count))))
    #take the logs after the multiplications
    spam_likelihood = np.log(spam_ratio) + np.log(spam_likelihood)
    ham_likelihood = np.log(ham_ratio) + np.log(ham_likelihood)

    if spam_likelihood > ham_likelihood:
      predictions[index1] = 1
  return predictions

train_features = pd.read_csv("sms_train_features.csv")
train_labels = pd.read_csv("sms_train_labels.csv")
test_features = pd.read_csv("sms_test_features.csv")
test_labels = pd.read_csv("sms_test_labels.csv")

train_features.drop(train_features.columns[0], axis=1, inplace=True)
test_features.drop(test_features.columns[0], axis=1, inplace=True)
train_labels.drop(train_labels.columns[0], axis=1, inplace=True)
test_labels.drop(test_labels .columns[0], axis=1, inplace=True)

#assign ham and spam counts
count_hams_and_spams(train_features)

#to calculate mutual information, the number of occurences of the word in spam or ham is stored in 4 arrays
hams_that_contain = np.zeros(size_of_vocabulary, dtype=int)
spams_that_contain = np.zeros(size_of_vocabulary, dtype=int)
hams_that_not_contain = np.zeros(size_of_vocabulary, dtype=int)
spams_that_not_contain = np.zeros(size_of_vocabulary, dtype=int)

mi_start = time.time()
#calculate mutual infos
for index1, train_sample in enumerate(train_features.itertuples(index=False, name='Pandas')):
  sample_label = int(train_labels.iloc[index1]) #indicates whether the message is spam
  if sample_label == 1:
    for index2, feature in enumerate(train_sample):
      if feature > 0:
        spams_that_contain[index2] += 1
      else:
        spams_that_not_contain[index2] += 1
  else:
    for index2, feature in enumerate(train_sample):
      if feature > 0:
        hams_that_contain[index2] += 1
      else:
        hams_that_not_contain[index2] += 1

mutual_infos = np.zeros(size_of_vocabulary, dtype=float)
mi = 0.0
total = spam_count + ham_count

for index in train_features.columns:
  i = int(index)
  total = spams_that_contain[i] + spams_that_not_contain[i] + hams_that_contain[i] + hams_that_not_contain[i]
  #print("sdc: ", spams_that_contain[i], "hdc:", hams_that_contain[i])
  s1 = (spams_that_contain[i] / total) * (np.log2((spams_that_contain[i] * total)/((spams_that_contain[i] + hams_that_contain[i]) * (spam_count))))
  s2 = (spams_that_not_contain[i] / total) * (np.log2((spams_that_not_contain[i] * total)/((total - spams_that_contain[i] - hams_that_contain[i]) * (spam_count))))
  s3 = (hams_that_contain[i] / total) * (np.log2((hams_that_contain[i] * total)/((spams_that_contain[i] + hams_that_contain[i]) * (ham_count))))
  s4 = (hams_that_not_contain[i] / total) * (np.log2((hams_that_not_contain[i] * total)/((total - spams_that_contain[i] - hams_that_contain[i]) * (ham_count))))

  if np.isnan(s1):
    s1 = 0
  if np.isnan(s2):
    s2 = 0
  if np.isnan(s3):
    s3 = 0
  if np.isnan(s4):
    s4 = 0

  mutual_infos[i] = s1 + s2 + s3 + s4


mi_end = time.time()
print("Mutual information calculation time:", mi_end-mi_start)

mutual_infos = enumerate(mutual_infos)
sorted_mutual_infos = sorted(mutual_infos, key=lambda i: i[1], reverse=True)

used_feature_indexes = np.zeros(size_of_vocabulary, int)
end = 100
start = 0

for i in range(6):
  #for index, item in sorted_mutual_infos[start:end]:
    #used_feature_indexes[index] = item

  train_begin = time.time()
  train(train_features, train_labels, sorted_mutual_infos, end)
  train_end = time.time()
  print("-------------------")
  print("Step", i + 1, "training time:", train_end - train_begin)

  predictions = predict(test_features, sorted_mutual_infos, end)

  number_of_guesses = 0
  number_of_correct_guesses = 0

  for index, item in enumerate(test_labels.itertuples(index=False, name='Pandas')):
      if(predictions[index] == item):
        number_of_correct_guesses += 1
      number_of_guesses += 1

  accuracy = number_of_correct_guesses/number_of_guesses
  print("Step", (i+1) ,"accuracy: ", accuracy)
  end += 100
  start += 100
