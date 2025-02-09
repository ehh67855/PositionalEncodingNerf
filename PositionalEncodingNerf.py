import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def positional_encoding(x, L=10):
    frequencies = 2.0 ** torch.arange(0, L, dtype=torch.float32) * np.pi
    x_expanded = x[..., None] * frequencies[None, :]
    encoding = torch.cat([torch.sin(x_expanded), torch.cos(x_expanded)], dim=-1)
    return encoding.view(x.shape[0], -1)

class NeRFModel(nn.Module):
    def __init__(self, input_dim=3, L=10, hidden_dim=128):
        super(NeRFModel, self).__init__()
        self.L = L
        self.input_dim = input_dim * (2 * L)
        
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = positional_encoding(x, self.L)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    return transform(img).permute(1, 2, 0)  # Convert to HWC format

def generate_rays(image):
    H, W, _ = image.shape
    i, j = torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing='ij')
    rays = torch.stack([i, j, torch.ones_like(i)], dim=-1).view(-1, 3)
    return rays

def train_nerf(model, image, epochs=1000, lr=1e-3):
    rays = generate_rays(image)
    target_colors = image.view(-1, 3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(rays)
        predicted_colors = output[:, 1:]
        loss = loss_fn(predicted_colors, target_colors)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def render_nerf(model, image):
    rays = generate_rays(image)
    output = model(rays)
    colors = output[:, 1:].view(image.shape)
    return colors.detach().numpy()

#temp data for now
image = load_image("images.jpeg")
model = NeRFModel(L=10)
train_nerf(model, image)
rendered_image = render_nerf(model, image)

# Display Rendered Image
plt.imshow(rendered_image)
plt.axis('off')
plt.show()
