import time

import matplotlib.pyplot as plt
import model
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm

from loaders.cifar_loader import train_loaders as cifar_loaders
from utils import plot_images

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# -------------------------hyperparameter---------------------------
num_epochs = 3
batch_train = 128  # reduce batch size if GPU run out memory
learning_rate = 0.01

random_seed = 5
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(random_seed)

# -----------------------loading the dataset------------------------
dataSet = "cifar"
# train_loader, val_loader = mnist_loaders(batch_train, random_seed, show_sample=True)
train_loader, val_loader = cifar_loaders(batch_train, random_seed, show_sample=True)
# ------------------------------------------------------------------

# ------------------------initialize network------------------------
modelName = "vgg16"
net = model.vgg16()
net.train().cuda()

# -----------------------------optimiser----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

start_ts = time.time()

losses = []
print(f"--------------------------------{dataSet}_{modelName}--------------------------------")
for epoch in range(num_epochs):
    total_loss = 0

    # progress bar
    progress = tqdm(enumerate(train_loader), desc="epoch: {} iter: {} Loss: {}", total=len(train_loader))
    # ----------------------------train--------------------------------
    for i, data in progress:
        # Data, Label
        X, y = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        losses.append(total_loss / (i + 1))

        # updating progress bar
        progress.set_description("epoch: {} iter: {} Loss: {:.4f}".format(epoch + 1, i, total_loss / (i + 1)))

    # ---------------------validation--------------------------
    correct = 0
    total = 0
    plot_img, plot_lab = [], []
    with torch.no_grad():
        for (images, labels) in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images).to(device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            while len(plot_img) < 9:
                plot_img = images[0:9].cpu().numpy()
                plot_lab = predicted[0:9].data

    plot_images(plot_img, plot_lab)

    print('Accuracy of the network on the %d test images: %f %%' % (total, 100 * correct / total))

# plotting
plt.plot(losses)

# save model to path
PATH = '../pretrained/{}_{}.pth'.format(dataSet, modelName)
torch.save(net.state_dict(), PATH)

print(f"Training time: {time.time() - start_ts}s")
plt.savefig("../plots/{}_{}_losses".format(dataSet, modelName))

plt.title(dataSet + "_" + modelName)
plt.xlabel('epoch')
plt.ylabel('log loss')
plt.show()
