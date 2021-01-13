import CNN_FC_model as model_set
import data_set as d_set
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 50

model = model_set.CNN3Net_200().to(DEVICE)
# model = model_set.Orin_CNN3Net_200().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

train_set = d_set.Indiapine_dataset(length=3200)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_set = d_set.Indiapine_dataset(length=800, start_pos=3300)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # loss = F.nll_loss(output, target)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)

        loss.backward()
        pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
        correct += pred.eq(target.view_as(pred)).sum().item()
        optimizer.step()
        if (batch_idx + 1) % 60 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrect: {:.0f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), 100. * correct / len(train_loader.dataset)))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

for epoch in range(1, EPOCH + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)

