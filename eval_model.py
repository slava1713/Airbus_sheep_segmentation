import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from training_script import train_loader



def show_results(model, loader, num_of_rows=16):
    fig, ax = plt.subplots(num_of_rows, 4, figsize=(16, 16))
    row = 0
    model = model.cpu()
    for image, mask in loader:
        #image = image.cpu().permute(0, 3, 1, 2)
        pred = model(image)
        m = nn.Softplus()
        pred = m(pred)
        image = image.cpu().permute(0, 2, 3, 1).detach().numpy()[0]
        mask = mask.cpu().detach()[0].numpy()
        pred = pred.detach()[0][0].cpu().numpy()
        ax[row][0].imshow(image)
        ax[row][1].imshow(mask)
        ax[row][2].imshow(pred)
        ax[row][3].imshow(image)
        ax[row][3].imshow(pred, alpha=0.5)
        for j in range(4):
            ax[row][j].axis('off')
        ax[row][0].set_title('image')
        ax[row][1].set_title('mask')
        ax[row][2].set_title('prediction')
        ax[row][3].set_title('image with prediction')
        row += 1
        if row == num_of_rows:
            break
    plt.savefig('evalimg.png')


path = "MyModel.pth"
model = torch.load(path)

show_results(model, train_loader, num_of_rows=8)

