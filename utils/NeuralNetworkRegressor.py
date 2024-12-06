import pathlib
import sys
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig
from utils.CodeLlamaPreprocessor import CodeLlamaPreprocessor

class NeuralNetworkRegressor(nn.Module):
    def __init__(self, out_path: pathlib.Path, input_size=256, hidden_layers=(128, 64, 32, 'd', 16, 8, 4), learning_rate=0.001, dropout_rate = 0.0, device='cpu', threshold=None):
        super(NeuralNetworkRegressor, self).__init__()

        if hidden_layers == 'large':
            hidden_layers = (128, 64, 32, 'd', 16, 8, 4)
        elif hidden_layers == 'small':
            hidden_layers = (128, 'd', 32, 8)

        self.device = device
        self.out_path = out_path
        self.best_result = None
        self.dropout_rate = dropout_rate
        self.input_size = input_size
        self.threshold = threshold
        
        # Create a list to hold the layers
        layers = []
        
        # Hidden layers
        prev_size = input_size
        for i, size in enumerate(hidden_layers):
            if size != 'd':
                layers.append(nn.Linear(prev_size, size))
                layers.append(nn.ReLU())
                prev_size = size
            else:
                layers.append(nn.Dropout(self.dropout_rate))
    
        size = 1
        
        # Output layer
        layers.append(nn.Linear(prev_size, size))
        layers.append(nn.Sigmoid())
        
        # Combine all layers into a sequential module
        self.network = nn.Sequential(*layers)
        
        # Loss function
        self.criterion = nn.MSELoss(reduction='none')
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(device)

    def get_preprocessor(self, model_path: str, **kwargs):
        return CodeLlamaPreprocessor(model_path, device = self.device, embedding_size = self.input_size, **kwargs)

    def forward(self, X_tensor):
        with torch.no_grad():
            return self.network(X_tensor)
    
    def fit(self, X_train, y_train, epochs=500, batch_size=32):
        # Calculate weights
        ispos = y_train > 0.5
        pos_prop = np.mean(np.where(ispos, 1, 0))
        neg_prop = 1 - pos_prop
        pos_weight = 1.0 / pos_prop
        neg_weight = 1.0 / neg_prop
        weights = np.where(ispos, pos_weight, neg_weight)

        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Convert input data to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # Create a DataLoader for batching
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.network(inputs)
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                weighted_loss = (loss * weights_tensor).mean()
                
                # Backward pass and optimize
                weighted_loss.backward()
                self.optimizer.step()
                
                epoch_loss += weighted_loss.item()

            # Print loss for the epoch
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}") ; sys.stdout.flush()

            if self.best_result is None or epoch_loss < self.best_result:
                self.best_result = epoch_loss
                self.best_model = None
                self.best_model = copy.deepcopy(self)
            
    
    def predict(self, X):
        # Convert input data to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.network(X_tensor)

        return predictions.cpu().numpy().flatten().reshape(X.shape[:-1])

    def predict_class(self, X):
        return self.predict(X) >= self.threshold

    def save(self, out_path = None):
        if out_path is None:
            out_path = self.out_path

        # Save the model's state dictionary
        torch.save(self.state_dict(), out_path)
        print(f"Model saved to {out_path}") ; sys.stdout.flush()

    @classmethod
    def load(cls, file_path, device='cpu', **kwargs):
        # Create an instance of the neural network
        model = cls(file_path, device=device, **kwargs)
        
        # Load the state dictionary from the file
        model.load_state_dict(torch.load(file_path, map_location=device))
        assert model.device == device
        
        # Set the model to evaluation mode
        model.to(model.device).eval()
        
        return model
