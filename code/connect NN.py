import torch
import torch.nn as nn

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # Define the architecture for the first model
        self.fc1 = nn.Linear(input_size1, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, output_size1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Define the forward pass of the first model
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        # Define the architecture for the second model
        self.fc1 = nn.Linear(input_size2, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, output_size2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Define the forward pass of the second model
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load pre-trained models
model1 = Model1()
model1.load_state_dict(torch.load('model1.pth'))
model1.eval()  # Set model to evaluation mode

model2 = Model2()
model2.load_state_dict(torch.load('model2.pth'))
model2.eval()  # Set model to evaluation mode

# Define input tensors
input1 = torch.randn(batch_size, input_size1)
input2 = torch.randn(batch_size, input_size2)

# Forward pass through both models
output1 = model1(input1)
output2 = model2(input2)

# Merge or combine outputs
merged = torch.cat((output1, output2), dim=1)

# Define additional layers if needed
merged = nn.Linear(merged.size(1), merged_output_size)(merged)
merged = nn.ReLU()(merged)

# Define output layer
output = nn.Linear(merged_output_size, num_classes)(merged)

# Create a combined model
combined_model = nn.Sequential(model1, model2, merged, output)

# Optionally, you can freeze the parameters of the pre-trained models
for param in combined_model.parameters():
    param.requires_grad = False

# Forward pass through the combined model
final_output = combined_model(input1, input2)
