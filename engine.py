""" Defines functions for training and testing a model.
"""
from typing import List, Dict, Tuple
import torch
from torch import nn

def train_step(dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               model: torch.nn.Module,
               device: torch.device,
               ):
  """ Train a PyTorch model for a single epoch.
  Args:
    model: A PyTorch model to be trained.
    loss_fn: A PyTprch loss function to calculate loss on train data.
    optimizer: A PyTorch optimizer.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    dataloader: A DataLoader instance for the model to be trained on.

  Returns:
    A tuple of training loss and training accuracy in the form: (training loss, training accuracy).
  """

  train_loss, train_acc = 0,0
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    model.train()
    
    train_pred = model(X)
    train_pred_labels = train_pred.argmax(dim=1)

    loss = loss_fn(train_pred, y)
    train_loss += loss
    train_acc += (train_pred_labels == y).sum().item()/len(y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
  train_loss /= len(dataloader)
  train_acc /= len(dataloader)
  
  return train_loss, train_acc

def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device):
  """ Test a PyTorch model for a single epoch.
  Args:
    model: A PyTorch model to be tested.
    loss_fn: A PyTprch loss function to calculate loss on test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    dataloader: A DataLoader instance for the model to be tested on.

  Returns:
    A tuple of tested loss and tested accuracy in the form: (tested loss, tested accuracy).
  """
  
  with torch.inference_mode():
    test_loss, test_acc = 0,0
    for batch, (X, y) in enumerate(dataloader):
      X, y = X.to(device), y.to(device)

      model.eval()

      test_pred = model(X)
      test_pred_labels = test_pred.argmax(dim=1)

      loss = loss_fn(test_pred, y)
      test_loss+= loss
      test_acc += (test_pred_labels == y).sum().item()/len(y)
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

  return test_loss, test_acc

from tqdm.auto import tqdm

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
  
          ):
  """ Train and test a PyTorch model for a given number of epochs.
  Args:
    model: A PyTorch model to be trained and tested.
    epochs: An integer indicating the number of iterations to be used for training and testing.
    loss_fn: A PyTprch loss function to calculate loss on train and test data.
    optimizer: A PyTorch optimizer.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.

  Returns:
    A dictionary containing the informations on the training and testing metrics (loss and accuracy).

  """

  results = { "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": [],
           }
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(train_dataloader, loss_fn, optimizer, model)
    test_loss, test_acc = test_step(model, test_dataloader, loss_fn)

    print(f'Epoch: {epoch} | Train loss: {train_loss:.4f} Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}')

    results['train_loss'].append(train_loss.item())
    results['train_acc'].append(train_acc)
    results['test_loss'].append(test_loss.item())
    results['test_acc'].append(test_acc)
  
  return results
