import json
from pathlib import Path
import numpy as np

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

path_to_training = Path("training")
path_to_test = Path("test")


#####
# training and test sets of transcription ids
#####
training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('IS1002a')
training_set.remove('IS1005d')
training_set.remove('TS3012c')

test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])


#####
# text_baseline: utterances are embedded with SentenceTransformer, then train a classifier.
#####

from sklearn.tree import DecisionTreeClassifier
from sentence_transformers import SentenceTransformer

bert = SentenceTransformer('all-MiniLM-L6-v2')

y_training = []
with open("training_labels.json", "r") as file:
    training_labels = json.load(file)
X_training = []
for transcription_id in training_set:
    with open(path_to_training / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
    
    for utterance in transcription:
        X_training.append(utterance["speaker"] + ": " + utterance["text"])
    
    y_training += training_labels[transcription_id]

X_training = bert.encode(X_training, show_progress_bar=True)


def train_and_predict_RF(B, m, X_train, y_train, X_test):
    """
    Inputs:
        B (int): number of trees
        m (int): number of features to be randomly chosen at each split in the tree
        X_train (Nxp matrix): training data
        y_train (N vector): response values
        X_test (N_test x p matrix): test data to be predicted
    Outputs:
        voting_test (N_test vector): 0-1 prediction for each individual in the test data
    """
    N_test = X_test.shape[0]
    N = X_train.shape[0]

    voting_test = np.zeros(N_test)
    model = DecisionTreeClassifier(max_features=m)
    for _ in range(B):
        bootstrap = np.random.choice(np.arange(N), size=N, replace=True)
        X_train_bootstrap = X_train[bootstrap]
        y_train_bootstrap = y_train[bootstrap]

        model.fit(X_train_bootstrap, y_train_bootstrap)
        voting_test += model.predict(X_test)

    voting_test = np.round(voting_test / B).astype(int)
    return voting_test

test_labels = {}
for transcription_id in test_set:
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
    
    X_test = []
    for utterance in transcription:
        X_test.append(utterance["speaker"] + ": " + utterance["text"])
    
    X_test = bert.encode(X_test)

    y_test = train_and_predict_RF(10, int(np.round(np.sqrt(X_training.shape[1]))), X_training, y_training, X_test)
    # y_test = clf.predict(X_test)
    test_labels[transcription_id] = y_test.tolist()

with open("test_labels_text_baseline.json", "w") as file:
    json.dump(test_labels, file, indent=4)
