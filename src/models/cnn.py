"""
cnn.py

2D CNN used for the Fashion-MNIST benchmark in FedBench.

Architecture (from the paper)

Input (1×28×28)

↓

Conv2D(6,5×5)

↓

ReLU

↓

MaxPool2D

↓

Conv2D(16,5×5)

↓

ReLU

↓

MaxPool2D

↓

Flatten (256)

↓

FC(256→120)

↓

ReLU

↓

FC(120→84)

↓

ReLU

↓

FC(84→10)
"""

import torch
import torch.nn as nn


class FashionMNISTCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2)

        )

        self.classifier = nn.Sequential(

            nn.Linear(256,120),

            nn.ReLU(),

            nn.Linear(120,84),

            nn.ReLU(),

            nn.Linear(84,10)

        )

    def forward(self,x):

        x=self.features(x)

        x=torch.flatten(x,1)

        x=self.classifier(x)

        return x


if __name__=="__main__":

    model=FashionMNISTCNN()

    print(model)

    x=torch.randn(1,1,28,28)

    y=model(x)

    print()

    print("Input Shape :",x.shape)

    print("Output Shape:",y.shape)