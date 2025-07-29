import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import time

# 设置随机种子确保结果可复现
torch.manual_seed(42)

# 定义数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# 加载CIFAR-100数据集
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 修改ResNet-50模型以适应CIFAR-100
model = resnet50(pretrained=False)
# 调整输入层以适应32x32图像
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
# 移除第一个最大池化层
model.maxpool = nn.Identity()
# 修改输出类别数为100
model.fc = nn.Linear(model.fc.in_features, 100)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练函数
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(f'Epoch: {epoch} | Train Loss: {train_loss/(batch_idx+1):.3f} | Train Acc: {100.*correct/total:.3f}%')

# 测试函数
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'Epoch: {epoch} | Test Loss: {test_loss/(batch_idx+1):.3f} | Test Acc: {100.*correct/total:.3f}%')
    return 100.*correct/total

# 训练模型
best_acc = 0.0
start_time = time.time()
for epoch in range(200):
    train(epoch)
    acc = test(epoch)
    scheduler.step()
    
    # 保存最佳模型
    if acc > best_acc:
        print(f'Saving model... Acc improved from {best_acc:.3f} to {acc:.3f}')
        torch.save(model.state_dict(), 'resnet50_cifar100.pth')
        best_acc = acc

end_time = time.time()
print(f'Training completed in {end_time - start_time:.2f} seconds')
print(f'Best accuracy: {best_acc:.3f}%')