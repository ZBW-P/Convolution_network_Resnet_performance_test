import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import torch.profiler
class resn_block(nn.Module):
    def __init__(self, input_channel,output_channel,kernek_size,stride,padding):
        super().__init__()
        self.block1=nn.Sequential(nn.Conv2d(input_channel,output_channel,kernek_size,stride,padding,bias=False),
                             nn.BatchNorm2d(output_channel),nn.ReLU())
        self.block2=nn.Sequential(nn.Conv2d(output_channel,output_channel,kernek_size,1,padding,bias=False),
                             nn.BatchNorm2d(output_channel))
        self.out_activation=nn.ReLU()
        
        self.idnety=nn.Identity()
        if input_channel != output_channel or stride != 1:
            self.trans_layer = nn.Conv2d(input_channel, output_channel,1,stride,0,bias=False)
        else:
            self.trans_layer = nn.Identity()
    def forward(self,x):
        X=self.idnety(x)
        x=self.block1(x)
        x=self.block2(x)
        x+=self.trans_layer(X)
        x=self.out_activation(x)
        return x
class resnet_18(nn.Module):
    def __init__(self, num_blocks,inpt_size,output_size,stride,kenerl_size,padding,class_label):
        super().__init__()
        self.int_conv=nn.Sequential(nn.Conv2d(inpt_size[0],output_size[0],kenerl_size[0],stride[0],padding[0],bias=False),nn.ReLU(),nn.BatchNorm2d(output_size[0]))
        self.modul=nn.Sequential()
        for n in range(num_blocks):
            input_channel=inpt_size[n+1]
            output_channel=output_size[n+1]
            kernel=kenerl_size[n+1]
            strid=stride[n+1]
            padd=padding[n+1]
            self.modul.add_module(f"resnet_block_{n+1}",
                                  resn_block(input_channel,output_channel,kernel,strid,padd))
        self.average=nn.AdaptiveAvgPool2d((1, 1))
        self.dens=nn.LazyLinear(class_label)
        
    def forward(self,x):
        X=self.int_conv(x)
        for block in self.modul:
            X=block(X)
        X=self.average(X)
        X = X.reshape(X.shape[0], -1) 
        X=self.dens(X)
        return X
    
def train(model: nn.Module, train_dataloader: DataLoader, optimizer,criterion, epochs=5, device="cuda"):

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch+1}/{epochs}")
        train_loss = 0
        correct = 0
        total = 0
        model.train()
        Load_time=0
        Train_time=0
        Epoch_time_start=time.perf_counter()
        for batch_idx, batch in enumerate(train_dataloader):
            Load_time_start=time.perf_counter()
            inputs, targets=batch

            Load_time_finish=time.perf_counter()
            Load_time+=Load_time_finish-Load_time_start
            
            Train_time_start=time.perf_counter()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            Train_time_end=time.perf_counter()
            Train_time+=Train_time_end-Train_time_start
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            batch_loss=loss.item()
            epoch_acc = 100.0 * correct / total
        Epoch_time_finish=time.perf_counter()
        Epoch_time=Epoch_time_finish-Epoch_time_start
        epoch_loss = train_loss / len(train_dataloader)
        epoch_acc = 100.0 * correct / total
        print(
            f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
        )
        print(
            f"Epoch_time: {Epoch_time:.8f}"
        )
        print(
            f"Load_time: {Load_time:.8f}"
        )
        print(
            f"Train_time: {Train_time:.8f}"
        )
def Train_with_profiler(model: nn.Module, train_dataloader: DataLoader, optimizer,criterion, epochs=5, device="cuda"):
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./Log2'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for epoch in range(epochs):
            print(f"\nEpoch: {epoch+1}/{epochs}")
            model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                with torch.profiler.record_function("data_loading"):
                    inputs, targets=batch
                    
                with torch.profiler.record_function("training_step"):
                    
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                prof.step()
    print("Saved the Log")
import argparse
def get_args():
    parser = argparse.ArgumentParser(description="--cuda --data_path --num_workers --batch_size --optimizer --epochs")
    parser.add_argument("--cuda", action="store_true", 
                        help="Use CUDA if available")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to dataset")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Mini-batch size")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd"],
                        help="Optimizer type")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--profiler", action="store_true",
                        help="If use profiler to record matrix")
    return parser.parse_args()

def train_loader(root,batch_size,worker):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])
    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=worker
    )
    return train_loader,test_loader
if __name__ == "__main__":
    args=get_args()
    
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model = resnet_18(
        num_blocks=8,
        inpt_size=[3,64,64,64,128,128,256,256,512],
        output_size=[64,64,64,128,128,256,256,512,512],
        stride=[1,1,1,2,1,2,1,2,1],
        kenerl_size=[3]*9,
        padding=[1]*9,
        class_label=10
    )
    model.to(device)
    epoch=args.epochs
    root=args.data_path
    worker=args.num_workers
    batch_size=args.batch_size
    if args.optimizer=='sgd':
        optim=torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    loss=nn.CrossEntropyLoss()
    train_dl, test_dl = train_loader(root, batch_size, worker)
    if args.profiler:
        Train_with_profiler(model, train_dl, optim, loss, epoch, device)
    else:
        train(model, train_dl, optim, loss, epoch, device)
