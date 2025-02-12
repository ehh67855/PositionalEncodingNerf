import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image  # Used to load and manipulate images

class NeRFModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super(NeRFModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 3)  # Output only RGB (no extra channels)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # No activation function for final output
        return x

def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    return transform(img).permute(1, 2, 0)  # Convert to HWC format

def generate_rays(image):
    H, W, _ = image.shape
    i, j = torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing='ij')
    rays = torch.stack([i, j, torch.ones_like(i)], dim=-1).view(-1, 3)  # (x, y, z=1)
    return rays

def train_nerf(model, image, epochs=5000, lr=1e-3):
    rays = generate_rays(image)  # Generate pixel coordinates
    target_colors = image.view(-1, 3)  # Flatten RGB values
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()  # Mean Squared Error Loss
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted_colors = model(rays)  # Predict colors from coordinates
        loss = loss_fn(predicted_colors, target_colors)  # Compare with true image
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def render_nerf(model, image):
    rays = generate_rays(image)
    output = model(rays)
    colors = output.view(image.shape)  # Reshape back into image dimensions
    return colors.detach().numpy()

# Load image
image = load_image("images.jpeg")

# Initialize and train the model
model = NeRFModel()
train_nerf(model, image)

# Render the trained image
rendered_image = render_nerf(model, image)

# Display Rendered Image
plt.imshow(rendered_image)
plt.axis('off')
plt.show()

