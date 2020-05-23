import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from loaders.cifar_loader import test_loader as cifar_loader
from resnet import model
from utils import print_scores, calculate_metric

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# -------------------------hyperparameter---------------------------
batch_test = 100

# -----------------------load model---------------------------------
kwargs = {
    "channel": 3
}
net = model.resnet50(pretrained=True, **kwargs)
net.eval().cuda()

# -------------------------optimiser---------------------------------
criterion = nn.CrossEntropyLoss()

# -----------------------loading the dataset------------------------
# test_loader = mnist_loader(batch_test)
test_loader = cifar_loader(batch_test)

if torch.cuda.is_available():
    # releasing unnecessary memory in GPU
    torch.cuda.empty_cache()
    # ----------------- TESTING  -----------------
    test_losses = 0
    precision, recall, f1, accuracy = [], [], [], []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            X, y = data[0].to(device), data[1].to(device)

            outputs = net(X)  # this get's the prediction from the network

            test_losses += criterion(outputs, y)

            predicted_classes = torch.max(outputs, 1)[1]  # get class from network's prediction

            # calculate P/R/F1/A metrics for batch
            for acc, metric in zip((precision, recall, f1, accuracy),
                                   (precision_score, recall_score, f1_score, accuracy_score)):
                acc.append(calculate_metric(metric, y.cpu(), predicted_classes.cpu()))

    # print(
    #     f"\nEpoch {epoch + 1}/{num_epochs}, training loss: {epoch / len(train_loader)}, validation loss: {test_losses / len(test_loader)}")
    print_scores(precision, recall, f1, accuracy, len(test_loader))
