import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import Callback
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torchvision
# You should implement these for CIFAR-10. HINT: The dataset may be accessed with Torchvision.
training_data = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((32, 32), antialias = True),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
# Batch Sizes = 64, 32, 16, 8
training_loader = DataLoader(training_data, batch_size = int(input("Enter Batch Size:")), shuffle = True, num_workers = 2)

# The cnn model class.
class EX_CNN_Module(pl.LightningModule):
    # Specify the structure of your model.
    def __init__(self, **kwargs):
        super(EX_CNN_Module, self).__init__()
        self.convolution_layers = nn.Sequential(
            # Revise to match CIFAR-10
            nn.Conv2d(in_channels = 3, out_channels = 32,
                    kernel_size = (3, 3), stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.BatchNorm2d(num_features = 32),

            nn.Conv2d(in_channels = 32, out_channels = 32,
                    kernel_size = (3, 3), stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.BatchNorm2d(num_features = 32),

            nn.Conv2d(in_channels = 32, out_channels = 32,
                    kernel_size = (3, 3), stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.BatchNorm2d(num_features = 32),
            
            nn.Conv2d(in_channels = 32, out_channels = 32,
                    kernel_size = (3, 3), stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.BatchNorm2d(num_features = 32),
            
            nn.Conv2d(in_channels = 32, out_channels = 32,
                    kernel_size = (3, 3), stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            nn.BatchNorm2d(num_features = 32)
        )

        self.linear_layers = nn.Sequential(
            # Revise to match CIFAR-10
            nn.Linear(in_features = 32, out_features = 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 64),

            nn.Linear(in_features = 64, out_features = 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 64),

            nn.Linear(in_features = 64, out_features = 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 64),

            nn.Linear(in_features = 64, out_features = 64),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = 64),

            nn.Linear(in_features = 64, out_features = 10),
            nn.Sigmoid()
        )
    
    # Define the data flow through your task graph.
    def forward(self, x):
        x = self.convolution_layers(x)
        x = torch.flatten(input = x, start_dim = 1)
        x = self.linear_layers(x)
        return x
    # Define optimizer.
    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=0.001)
        return optimizer
    # Define training step.
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = self(inputs)
        loss = criterion(outputs, labels)

        #self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        if batch_idx % 20 == 19:
            # Print running loss.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                self.current_epoch, batch_idx * len(inputs), len(training_loader.dataset),
                100. * batch_idx / len(training_loader), loss.item()))
        return loss
    
# Instantiate your model.
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Create the LightningModule instance
model = EX_CNN_Module()

# Define criterion
criterion = nn.CrossEntropyLoss()

# Create the Trainer instance
trainer = pl.Trainer(max_epochs = 2, accelerator = 'auto')

# Train the model
trainer.fit(model, training_loader)
print("Finished training.")
