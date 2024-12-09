import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from PianoRollProcessor import PianoRollProcessor
from typing import Tuple, List, Optional
from MusicLSTM import MusicLSTM

class PianoRollDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        print(f"Created dataset with {len(self.X)} samples")
        print(f"Sample X: {self.X[0]}")  # Show sample input data
        print(f"Sample y: {self.y[0]}")  # Show sample target data
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MusicTrainer:
    def __init__(self, processor: PianoRollProcessor, sequence_length: int = 100,
                 hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2, learning_rate: float =0.001):
        self.processor = processor
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nInitializing MusicGenerator:")
        print(f"Device: {self.device}")
        print(f"Sequence length: {sequence_length}")
        print(f"Hidden size: {hidden_size}")
        print(f"Number of LSTM layers: {num_layers}")
        
        self.model = MusicLSTM(
            input_size=processor.note_range,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        self.training_data = None

    def train(self, midi_folder: str, num_epochs: int = 1, batch_size: int = 64,
              learning_rate: float = 0.001, save_path: str = 'model.pth'):
        print(f"\nLoading MIDI files from: {midi_folder}")
        
        all_piano_rolls = []
        midi_files = [f for f in os.listdir(midi_folder) if f.endswith('.mid')]
        print(f"Found {len(midi_files)} MIDI files")
        
        for i, midi_file in enumerate(midi_files, 1):
            print(f"\nProcessing file {i}/{len(midi_files)}: {midi_file}")
            piano_roll, duration = self.processor.midi_to_piano_roll(os.path.join(midi_folder, midi_file))
            all_piano_rolls.append(piano_roll)
            print(f"Piano roll shape: {piano_roll.shape}")
        
        self.training_data = np.concatenate(all_piano_rolls, axis=0)
        print(f"\nCombined training data shape: {self.training_data.shape}")
        
        X, y = self.processor.prepare_sequences(self.training_data, self.sequence_length)
        print(f"Prepared sequences - X shape: {X.shape}, y shape: {y.shape}")
        print(f"Sample X sequence: {X[0]}")  # Sample sequence input
        print(f"Sample y sequence: {y[0]}")  # Sample sequence target
        
        dataset = PianoRollDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"Created DataLoader with {len(dataloader)} batches")
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        print(f"\nStarting training for {num_epochs} epochs:")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size}")
        
        best_loss = float('inf')
        best_model_state = None  # To store the best model state

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader, 1):
                if batch_idx % 10 == 0:  # Print progress every 10 batches
                    print(f"Processing batch {batch_idx}/{len(dataloader)}", end='\r')
                
                print(f"Batch X shape: {batch_X.shape}")  # Show batch X shape
                print(f"Batch y shape: {batch_y.shape}")  # Show batch y shape
                
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                hidden = self.model.init_hidden(batch_X.size(0), self.device)
                optimizer.zero_grad()
                
                output, _ = self.model(batch_X, hidden)
                loss = criterion(output, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch+1} completed - Average loss: {avg_loss:.4f}")
            
            # Save model if it is the best one so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = self.model.state_dict()  # Save the model's state
                print(f"New best model found, saving model...")

            # Save model at the end of each epoch with loss information
            model_filename = f"{save_path}_epoch{epoch+1}_loss{avg_loss:.4f}.pth"
            torch.save(self.model.state_dict(), model_filename)
            print(f"Model saved to {model_filename}")
        
        # Finally, save the best model at the end of training
        if best_model_state is not None:
            best_model_filename = f"{save_path}_best_model.pth"
            torch.save(best_model_state, best_model_filename)
            print(f"Best model saved to {best_model_filename}")
