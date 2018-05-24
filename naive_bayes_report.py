import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import KFold


#Gives the prior probabilities of label
def label_prior_probabilites(data):
    samples = len(data)
    label_list = []
    for i in range(samples):
        label_list.append(data[i][-1])
    counter = dict()
    for i in label_list:
        if not i in counter:
            counter[i] = 1
        else:
            counter[i] += 1
    for key in counter.keys():
        counter[key] = float(counter[key])/ float(samples)
    return counter

#This function seperates the data based on the label
def seperate(data):
    seperated_data = defaultdict(list)
    for sample in data:
        seperated_data[sample[-1]].append(sample)
    new_list = seperated_data.values()
    no_of_labels = len(new_list)
    data_dict = dict()
    for i in range(no_of_labels):
        data_dict[i] = new_list[i]
    return data_dict

#Performs naive bayes and classifies it to a label
def naive_bayes(data,testing_set):
        rows = len(data)
        cols = len(data[0])
        labels = []
        for i in range(rows):
            labels.append(data[i][-1])
        unique_labels = np.unique(labels)
        label_probabiltites = label_prior_probabilites(data)
        data_by_class = seperate(data)
        length = len(unique_labels)
        i = 0
        result = []
        for sample in testing_set:

            prob_dr = 1.0
            temp = [0] * (cols -1)
            temp_2 = [0] * (cols -1)
            i = 0
            for find_value in data:
                for i in range(len(sample) - 1):
                    if find_value[i] == sample [i]:
                        temp_2[i] = temp_2[i] + 1
            #print rows
            #print temp_2
            for j in range(len(temp_2)):
                prob_dr = prob_dr * ((float(temp_2[j])/float(rows))+1)
            answer =dict()
            maximum = 999
            i =0
            for label in data_by_class:
                prob_nr = 1.0
                prob = 1.0
                for value in data_by_class[label]:
                    for k in range(len(sample) - 1):
                        if value[k] == sample[k]:
                            temp[k] = temp[k] +1
                size = len(data_by_class[label])
                print temp
                print size
                for t in range(len(sample) - 1):
                    #print temp[t]
                    #print size
                    prob = float(prob) * ((float(temp[t])/float(size))+1)
                #print prob_dr
                print "prob",prob
                prob =     float(label_probabiltites[label] * prob)/float(prob_dr)
                print "prob_temp",prob
                answer[i] = prob
                i = i + 1
            print answer
            for key in answer:
                if answer[key] < maximum:
                    maximum = answer[key]
                    label_chosen = key
            result.append(label_chosen)
        print result
        return result




def performace_param(predicted_labels_ret, labels_testing_set):

    tp = 0

    tn = 0

    fp = 0

    fn = 0

    for i in range(len(predicted_labels_ret)):

        if(predicted_labels_ret[i] == labels_testing_set[i] == 1):
            tp = tp + 1
        elif(predicted_labels_ret[i] == labels_testing_set[i] == 0):
            tn = tn + 1
        elif(predicted_labels_ret[i] == 1 and labels_testing_set[i] == 0):
            fp = fp + 1
        elif(predicted_labels_ret[i] == 0 and labels_testing_set[i] == 1):
            fn = fn + 1

    return tp, tn, fp, fn


def calculate_accuracy(predicted_labels_ret, labels_testing_set):

    tp, tn, fp, fn = performace_param(predicted_labels_ret, labels_testing_set)
    if((tp + tn + fp + fn) == 0):
        return float(tp + tn)
    accuracy = float(tp + tn)/float(tp + tn + fp + fn)

    return accuracy


def calculate_precision(predicted_labels_ret, labels_testing_set):

    tp, tn, fp, fn = performace_param(predicted_labels_ret, labels_testing_set)
    if((tp + fp) == 0):

        return float(tp)

    precision = float(tp)/float(tp + fp)

    return precision


def calculate_recall(predicted_labels_ret, labels_testing_set):

    tp, tn, fp, fn = performace_param(predicted_labels_ret, labels_testing_set)
    if((tp + fn) == 0):

        return float(tp)
    recall = float(tp)/float(tp + fn)

    return recall



def calculate_fmeasure(predicted_labels_ret, labels_testing_set):

    tp, tn, fp, fn = performace_param(predicted_labels_ret, labels_testing_set)
    if((2 * tp + fp + fn) == 0):

        return float(2 * tp)
    fmeasure = float(2 * tp)/float(2 * tp + fp + fn)

    return fmeasure
#step 1 - format input data

print " "
result_Data = []
with open('project3_dataset2.txt','r') as f:
    for line in f:
        result_Data.append(line.strip().split('\t'))
rows = len(result_Data)
cols = len(result_Data[0])

for i in range(rows):
    for j in range(cols):
        if not(result_Data[i][j].isalpha()):
            result_Data[i][j] = float(result_Data[i][j])

for i in range(rows):
    result_Data[i][-1] = int(result_Data[i][-1])

result_Data = np.array(result_Data, dtype = object)

data_from_file = result_Data

labels = np.genfromtxt("project3_dataset2.txt", usecols = -1, dtype = None)

splits = 10

kf = KFold(n_splits=10)

kf.get_n_splits(data_from_file)

total_accuracy = 0

total_precision = 0

total_recall = 0

total_fmeasure = 0

for train_index, test_index in kf.split(data_from_file):

    training_set, testing_set = data_from_file[train_index], data_from_file[test_index]

    labels_training_set, labels_testing_set = labels[train_index], labels[test_index]

    predicted_labels_ret = naive_bayes(training_set, testing_set)

    accuracy = calculate_accuracy(predicted_labels_ret, labels_testing_set)

    total_accuracy = total_accuracy + accuracy

    precision = calculate_precision(predicted_labels_ret, labels_testing_set)

    total_precision = total_precision + precision

    recall = calculate_recall(predicted_labels_ret, labels_testing_set)

    total_recall = total_recall + recall

    f1_measure = calculate_fmeasure(predicted_labels_ret, labels_testing_set)

    total_fmeasure = total_fmeasure + f1_measure

    # print accuracy , precision, recall, f1_measure
    #
    # print predicted_labels_ret

print float(total_accuracy )/ float(splits)

print float(total_precision )/ float(splits)

print float(total_recall )/ float(splits)

print float(total_fmeasure)/ float(splits)
