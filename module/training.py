import json

from pre_processing import path_to_test, test_set
from embedding import transformer, X_training, y_training
from model import *

X_training_tensor = torch.tensor(X_training, dtype=torch.float32)
y_training_tensor = torch.tensor(y_training, dtype=torch.float32)


input_size = X_training_tensor.size(1)
model = BinaryClassificationModel(input_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_training_tensor), batch_size):
        inputs = X_training_tensor[i:i+batch_size]
        labels = y_training_tensor[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward()
        optimizer.step()



# Solution
test_labels = {}
for transcription_id in test_set:
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
    
    X_test = []
    for utterance in transcription:
        X_test.append(utterance["speaker"] + ": " + utterance["text"])
    
    X_test = transformer.encode(X_test)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs >= 0.5).float()

    test_labels[transcription_id] = predictions.tolist()

with open("test_labels_RNN.json", "w") as file:
    json.dump(test_labels, file, indent=4)
