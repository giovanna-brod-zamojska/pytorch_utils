
import os 
import torch
from torchvision import transforms
import data_setup, engine, utils, model_builder
from timeit import default_timer as timer
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

train_dir = 'data/pizza_steak_sushi/train'
test_dir = 'data/pizza_steak_sushi/test'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                    test_dir=test_dir,
                                                                    transform=data_transform,
                                                                    batch_size=BATCH_SIZE,
                                                                    )

model = model_builder.TinyVGG(input_shape=3,
                output_shape=len(class_names),
                hidden_units= HIDDEN_UNITS
                )

optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

start_time = timer()
engine.train(model=model,
      test_dataloader=test_dataloader,
      train_dataloader=train_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      epochs=NUM_EPOCHS,
      device=device)
end_time()
print(f'[INFO] Total training time: { end_time - start_time} seconds')

utils.save_model(model=model, target_dir='models',model_name='TinyVGGV0.pt')
