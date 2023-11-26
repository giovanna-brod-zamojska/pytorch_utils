import requests
import os
import zipfile
from pathlib import Path


def download_data(data_folder_name:  str,
                  url: str):
  """ Download a zip file from a given url. Extracts the data 
  in a folder named 'data/data_folder_name', and then removes the '.zip' file.
  The data_folder_name should be the same as the zip file (e.g. 'data_folder_name.zip').
  The url should point to the zip file (e.g.  https://github.com/path/to/zipfile.zip)

  Args: 
    data_folder_name: A string indicating the name of the destination folder. It will be created within the 'data/' folder.
      It should coincide with the name of the zip folder.
    url: A string url indicating the path/to/the/zip/file.zip

  """
  data_path = Path('data/')
  image_path = data_path / data_folder_name

  if not image_path.is_dir():
    image_path.mkdir(parents=True, exist_ok=True )

  with open(data_path / f'{data_folder_name}.zip', 'wb') as f:
    request = requests.get(url)
    f.write(request.content)

  with zipfile.ZipFile(data_path / f'{data_folder_name}.zip', 'r') as zip_ref:
    zip_ref.extractall(image_path)

  os.remove(data_path / f'{data_folder_name}.zip')


