import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import UperNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import matplotlib.pyplot as plt

# Define the dataset
class CustomDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.frame_paths = []
        self.transform = transform
        for root_dir in root_dirs:
            self.frame_paths.extend(sorted(glob.glob(os.path.join(root_dir, '*.png'))))

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        frame = Image.open(frame_path).convert('RGB')

        idx, thumb_pos, target = os.path.basename(frame_path)[:-4].split('_')
        thumb_pos = torch.Tensor(list(eval(thumb_pos)))
        target = torch.Tensor([int(target)])

        if self.transform:
            frame = self.transform(frame)

        # Binarize the target using a threshold
        threshold = 150  # You can adjust this threshold based on dataset
        target = torch.clip(target, 0, 300)
        target = (target >= threshold).float()

        return frame, thumb_pos, target

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Main execution code
if __name__ == '__main__':
    # List of directories

    train_root_dirs = []
    test_root_dirs = []
    # for i in range(0, 5):
    #     for j in range(1, 4):
    #             train_root_dirs.append(
    #                 f'data/P000, CNA, phase0, Positive, trialSession-{i}, 180s, egocamera{j}_original/'
    #             )

    # test_root_dirs = [
    #     'data/P000, CNA, phase0, Test, trialSession-5, 180s, egocamera1_original/',
    #     'data/P000, CNA, phase0, Test, trialSession-5, 180s, egocamera2_original/',
    #     'data/P000, CNA, phase0, Test, trialSession-5, 180s, egocamera3_original/',
    # ]
    
    json_file_paths, csv_file_paths, video_paths, output_dirs = [], [], [], []
    action_label = ['CPinch','CSide1','CSide2','CSide3']
    # train_force_label = ['Negative1','Negative2','Positive1','Positive2','Positive3','Positive4']
    # test_force_label = ['Negative3','Positive5','Positive6']
    train_force_label = ['Negative1','Negative2','Positive2','Positive4']
    test_force_label = ['Negative3','Positive6']
    phase_label = ['phase0','phase1']
    # Train data
    for condition in action_label:
        for force in train_force_label[:2]:
            for k in range(1, 4):
                for j in range(1, 4):
                    train_root_dir = f"data/hsj/P000, {condition}, phase0, {force}, trialSession-{k}, 5s, egocamera{j}_original/"
                    train_root_dirs.append(train_root_dir)
    
    for condition in action_label:
        for force in train_force_label[2:]:
            for k in range(1, 4):
                for j in range(1, 4):
                    train_root_dir = f"data/hsj/P000, {condition}, phase1, {force}, trialSession-{k}, 20s, egocamera{j}_original/"
                    train_root_dirs.append(train_root_dir)
    # Test data
    for condition in action_label:
        for force in test_force_label[0]:
            for k in range(1, 4):
                for j in range(1, 4):
                    test_root_dir = f"data/hsj/P000, {condition}, phase0, {force}, trialSession-{k}, 5s, egocamera{j}_original/"
                    test_root_dirs.append(test_root_dir)
    
    for condition in action_label:
        for force in test_force_label[1:]:
            for k in range(1, 4):
                for j in range(1, 4):
                    test_root_dir = f"data/hsj/P000, {condition}, phase1, {force}, trialSession-{k}, 20s, egocamera{j}_original/"
                    test_root_dirs.append(test_root_dir)
    

    # Create dataset
    train_dataset = CustomDataset(root_dirs=train_root_dirs, transform=transform)
    test_dataset = CustomDataset(root_dirs=test_root_dirs, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UperNet(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, thumb_pos, targets in train_loader:
            inputs = inputs.to(device)
            thumb_pos = thumb_pos.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            batch_size, _, height, width = outputs.shape

            # Denormalize thumb_pos to match the spatial dimensions of the outputs
            x_coords = torch.round(thumb_pos[:, 0] // 8).long()
            y_coords = torch.round(thumb_pos[:, 1] // 8).long()

            # Ensure coordinates are within the output dimensions
            x_coords = x_coords.clamp(0, width - 1)
            y_coords = y_coords.clamp(0, height - 1)

            # Gather the output values at the specified coordinates
            selected_outputs = outputs[torch.arange(batch_size), 0, y_coords, x_coords]

            loss = criterion(selected_outputs, targets.squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # Evaluation
        if (epoch + 1) % 5 != 0:
            continue

        model.eval()
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, thumb_pos, targets in test_loader:
                inputs = inputs.to(device)
                thumb_pos = thumb_pos.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                batch_size, _, height, width = outputs.shape

                x_coords = torch.round(thumb_pos[:, 0] // 8).long()
                y_coords = torch.round(thumb_pos[:, 1] // 8).long()

                # Ensure coordinates are within the output dimensions
                x_coords = x_coords.clamp(0, width - 1)
                y_coords = y_coords.clamp(0, height - 1)

                selected_outputs = outputs[torch.arange(batch_size), 0, y_coords, x_coords]

                probs = torch.sigmoid(selected_outputs)
                preds = (probs >= 0.5).float()

                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())

        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)

        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)

        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
