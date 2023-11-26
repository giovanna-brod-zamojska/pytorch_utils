
""" 
File containing various utility functions for PyTorch model traning.
"""

from pathlib import Path
import torch

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """ Save a PyTorch model (state dict) to a target directory.

  Args:
    model: A PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A string indicating the filename for the saved model. 
      Should include either '.pth' or '.pt' as file extensions.


  """

  model_path = Path(target_dir)
  model_path.mkdir(parents=True, exist_ok=True)
  assert model_name.endswith('.pth') or model_name.endswith('.pt'), "model_name should end either with '.pth' or '.pt'."
  model_save_path = model_path / model_name
  torch.save(obj=model.state_dict(), f=model_save_path)
  print(f'Model saved to {model_save_path}.')
