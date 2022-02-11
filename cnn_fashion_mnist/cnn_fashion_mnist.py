import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../fashion_mnist_cnn/runs/fashion_mnist_cnn')

# training dataset download
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# test dataset download
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# generating dataLoader
train_dataloader = DataLoader(training_data, batch_size=100)
test_dataloader = DataLoader(test_data, batch_size=100)

# GPU 이용가능하면 GPU로 돌리자
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} devise")


# model define
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer = nn.Sequential(
            # [100, 1, 28, 28] -> [100, 16, 24, 24]
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),

            # [100, 16, 24, 24] -> [100, 32, 20, 20]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),

            # [100, 32, 20, 20] -> [100, 32, 10, 10]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # [100, 32, 10, 10] -> [100, 64, 6, 6]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),

            # [100, 64, 6, 6] -> [100, 64, 3, 3]
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layer = nn.Sequential(
            # [100, 64 * 3 * 3] -> [100, 100]
            nn.Linear(64 * 3 * 3, 100),
            nn.ReLU(),
            # [100, 100] -> [100, 10]
            nn.Linear(100, 10)
        )

    def forward(self, x):
        # self.layer에 정의한 연산 수행
        out = self.layer(x)

        # view 함수를 이용해 텐서의 형태응 [100, 나머지]로 변환
        out = out.view(100, -1)

        # self.fc_layer 정의한 연산 수행
        out = self.fc_layer(out)
        return out

model = CNN().to(device)
print(model)

# parameter optimazation
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9) # 0.001

# 각 학습 단계(training loop)에서 모델은 배치로 제공되는 학습 데이터셋에 대한 예측을 수행하고,
# 예측 오류를 역전파하여 모델의 매개변수를 조정한다.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    avg_loss = 0
    num_batch = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # GPU로 업로드되어서 사용된다는 뜻

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 평균 loss 계산
        avg_loss += loss_fn(pred, y).item()

        # 역전파
        optimizer.zero_grad()
        loss.backward() # loss에 대한 W의 변화량, b의 변화량 계산
        optimizer.step()

        if batch % 100 == 0:
            loss,  current = loss.item(), batch * len(X)
            print(f"loss : {loss:>7f} [{current:>5d}/{size:>5d}]")

    avg_loss /= num_batch
    return avg_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() #
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct

for epoch in range(10):
    print(f"Epoch {epoch+1}\n--------------------------------------")
    training_loss = train(train_dataloader, model, loss_fn, optimizer)
    writer.add_scalar("Loss/train", training_loss, epoch)
    testing_loss, accuracy = test(test_dataloader, model, loss_fn)
    writer.add_scalar("Loss/test", testing_loss, epoch)
    writer.add_scalar("Accuracy/test", accuracy, epoch)
writer.close()
print("Done!")

# # saving model
# # 모델을 저장하는 일반적인 방법은 (모델의 매개변수들을 포함하여) 내부 상태 사전을 직렬화하는 것이다.
# torch.save(model.state_dict(), "../model_fashion_mnist.pth")
# print("Saved PyTorch Model State to model_fashion_mnist.pth")
#
# # loading model
# model = NeuralNetwork()
# model.load_state_dict(torch.load("model_fashion_mnist.pth"))



