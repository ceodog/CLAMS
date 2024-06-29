import os
import pandas as pd
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTModel, ViTConfig
from sklearn.metrics import accuracy_score, classification_report


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to calculate the total memory allocated on CUDA device
def get_memory_allocated():
    return torch.cuda.memory_allocated(device=device)

# Function to print memory usage summary
def print_memory_summary():
    print(torch.cuda.memory_summary(device=device))

class Encoder(nn.Module):
    def __init__(self, vit_config, device=torch.device("cpu")):
        super().__init__()
        self.vit_config = vit_config
        self.vit = ViTModel(ViTConfig(**vit_config))
        self.classifier = nn.Linear(vit_config['hidden_size'], vit_config['num_classes'])
        self.device = device

    def forward(self, data):
        outputs = self.vit(data)  # Assuming spc is the spectroscopy data
        last_hidden_state = outputs.last_hidden_state[:, 0]  # Only use the first token ([CLS]) representation
        logits = self.classifier(last_hidden_state)
        return logits

    def load_weights(self, model_dir):
        weight_file = os.path.join(model_dir, 'model_weights.pth')
        self.load_state_dict(torch.load(weight_file, map_location=self.device))

    def save_weigths(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        weight_file = os.path.join(model_dir, 'model_weights.pth')
        torch.save(self.state_dict(), weight_file)

    def test_model(self, model_dir=None, test_loader=None, device=torch.device('cpu'),
                        labels=None):
        try:
            self.load_weights(model_dir)
            logging.info("Model weights loaded from %s! Calculating metrics...", model_dir)
        except Exception as ex:
            logging.error("Error loading model weights: %s", ex)

        img_size = self.vit_config['image_size']

        self.to(device)
        self.eval()

        # Lists to store true labels and predicted labels
        true_labels_list = []
        predicted_labels_list = []

        # Iterate over test data loader
        for batch in tqdm(test_loader):
            label, smile, img, ir, uv, nmr = batch
            data = img.view(-1, 1, img_size[0], img_size[1]).to(device)

            # Forward pass
            with torch.no_grad():
                embeddings = self(data)

            # Convert logits to binary predictions using threshold of 0.5
            predicted_labels = torch.sigmoid(embeddings) > 0.5

            # Convert predictions and labels to CPU and numpy arrays for comparison
            predicted_labels = predicted_labels.cpu().numpy()
            true_labels = label.numpy()

            # Append true labels and predicted labels to lists
            true_labels_list.extend(true_labels)
            predicted_labels_list.extend(predicted_labels)

        # Convert lists to numpy arrays
        true_labels_array = np.array(true_labels_list)
        predicted_labels_array = np.array(predicted_labels_list)

        # Calculate accuracy
        accuracy = accuracy_score(true_labels_array, predicted_labels_array)

        # Generate classification report
        report = classification_report(true_labels_array, predicted_labels_array, \
                        target_names=labels)
                        
        report_v2 = classification_report(true_labels_array, predicted_labels_array, \
                target_names=labels, output_dict=True)
                
        report_df = pd.DataFrame(report_v2).transpose()

        logging.info("Accuracy: %f", accuracy)
        logging.info("\nClassification Report:\n%s", report)
        return {"accuracy_score": accuracy, "classification_report": report, "report_df": report_df}

    def train_model(self, num_epochs=100, lr=1e-3, step_size=1, gamma=0.975, early_stopping_epochs=6,
                  model_dir=None, train_loader=None, val_loader=None, device=torch.device('cpu')):

        assert model_dir is not None, "Please supply model folder to save trained model weights!!!"

        self.to(device)
        img_size = self.vit_config['image_size']

        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        min_val_loss = float('inf')
        patience = 0

        metrics = {'epoch': [], 'train_loss': [], 'val_loss': []}
        loss_function = torch.nn.MultiLabelSoftMarginLoss()

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for batch in tqdm(train_loader):
                label, smile, img, ir, uv, nmr = batch
                data = img.view(-1, 1, img_size[0], img_size[1]).to(device)
                optimizer.zero_grad()
                embeddings = self(data)
                loss = loss_function(embeddings, label.float().to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)

            # Calculate average training loss
            train_loss /= len(train_loader.dataset)
            metrics['train_loss'].append(train_loss)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}")

            # Validation
            self.eval()
            val_loss = 0.0
            for batch in val_loader:
                label, smile, img, ir, uv, nmr = batch
                data = img.view(-1, 1, img_size[0], img_size[1]).to(device)
                embeddings = self(data)
                loss = loss_function(embeddings, label.float().to(device))
                val_loss += loss.item() * data.size(0)

            # Calculate average validation loss
            val_loss /= len(val_loader.dataset)
            metrics['val_loss'].append(val_loss)

            logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")

            # Early stopping
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                patience = 0
                logging.info("Saving model weights...")
                self.save_weigths(os.path.join(model_dir))
            else:
                patience += 1

            if patience >= early_stopping_epochs:
                logging.info("Early stopping triggered.")
                break

            # Adjust learning rate
            scheduler.step()
