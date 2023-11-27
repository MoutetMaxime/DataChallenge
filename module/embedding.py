import json
from sentence_transformers import SentenceTransformer

from pre_processing import path_to_training, training_set

transformer = SentenceTransformer('all-MiniLM-L6-v2')

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

X_training = transformer.encode(X_training, show_progress_bar=True)