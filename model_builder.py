"""
Contains PyTorch model code to instantiate a TinyVGG architecture.
"""
import torch
from torch import nn

class TinyVGG(nn.Module):
  """ Creates a TinyVGG architecture, replicated from: https://poloclub.github.io/cnn-explainer/.
  
  Args:
    input_shape: An integer indicating the number of input channels.
    outpute_shape: An integer indicating the number of output channels.
    hidden_units: An integer indicating the number of hidden untis between layers.
  """

  def __init__(self,
                  input_shape: int,
                  hidden_units: int,
                  output_shape: int
                  ):
    super().__init__()
    self.conv_block1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, 
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  stride=1,
                  padding=1,
                  kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv_block2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, 
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  stride=1,
                  padding=1,
                  kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*16*16,
                  out_features=output_shape)
    )

  def forward(self,x):
    return self.classifier(self.conv_block2(self.conv_block1(x)))
    

