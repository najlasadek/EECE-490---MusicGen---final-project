import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pretty_midi

# Class Definitions

class PianoRollProcessor:
    def __init__(self, lowest_note: int = 21, highest_note: int = 108, fs: int = 4):
        self.lowest_note = lowest_note
        self.highest_note = highest_note
        self.note_range = highest_note - lowest_note + 1  # Should be 88
        self.fs = fs
        print(f"Initialized processor with note range: {self.note_range} notes ({lowest_note}-{highest_note})")
        print(f"Sampling rate: {fs}Hz ({fs} steps per second)")
        assert self.note_range == 88, "Error: The note range is not 88. Check 'lowest_note' and 'highest_note' settings."

    def midi_to_piano_roll(self, midi_path: str):
        """
        Converts a MIDI file to a piano roll representation.
        """
        print(f"\nProcessing MIDI file: {midi_path}")
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        duration = midi_data.get_end_time()
        total_steps = int(duration * self.fs)
        piano_roll = np.zeros((total_steps, self.note_range))
        print(f"Initialized piano roll with dimensions: {piano_roll.shape} (timesteps x notes)")

        note_count = 0
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    start_idx = int(note.start * self.fs)
                    end_idx = int(note.end * self.fs)
                    note_idx = note.pitch - self.lowest_note
                    
                    if 0 <= note_idx < self.note_range:
                        piano_roll[start_idx:end_idx, note_idx] = 1
                        note_count += 1
                    else:
                        print(f"Note {note.pitch} out of range and ignored.")

        assert piano_roll.shape[1] == 88, f"Error: Piano roll has {piano_roll.shape[1]} notes instead of 88."
        print(f"Final piano roll dimensions: {piano_roll.shape} (timesteps x 88)")
        print(f"Duration: {duration:.2f}s, Total steps: {total_steps}, Total notes: {note_count}")
        return piano_roll, duration

    def prepare_sequences(self, piano_roll: np.ndarray, sequence_length: int):
        """
        Prepares sequences for LSTM training from the piano roll.
        """
        print(f"\nPreparing sequences with length {sequence_length}")
        X, y = [], []
        
        for i in range(len(piano_roll) - sequence_length):
            sequence = piano_roll[i:i + sequence_length]
            target = piano_roll[i + sequence_length]
            X.append(sequence)
            y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        print(f"Prepared {len(X)} sequences")
        return X, y

class MusicLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(MusicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        out = self.fc(lstm_out[:, -1, :])  # Take only the last time step
        out = self.sigmoid(out)
        return out, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))

class MusicTrainer:
    def __init__(self, processor: PianoRollProcessor, sequence_length: int = 100, hidden_size: int = 512, 
                 num_layers: int = 4, dropout: float = 0.2):
        self.processor = processor
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.model = MusicLSTM(input_size=processor.note_range, hidden_size=hidden_size, 
                               num_layers=num_layers, dropout=dropout).to(self.device)

    def train(self, midi_folder: str, num_epochs: int, batch_size: int, learning_rate: float, save_dir: str, start_epoch: int = 0):
        print(f"\nLoading MIDI files from: {midi_folder}")
        
        all_piano_rolls = []
        midi_files = [f for f in os.listdir(midi_folder) if f.endswith('.mid')]
        for midi_file in midi_files:
            piano_roll, _ = self.processor.midi_to_piano_roll(os.path.join(midi_folder, midi_file))
            all_piano_rolls.append(piano_roll)
        
        training_data = np.concatenate(all_piano_rolls, axis=0)
        print(f"\nTraining data shape: {training_data.shape}")
        
        X, y = self.processor.prepare_sequences(training_data, self.sequence_length)
        dataset = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(y))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        best_loss = float('inf')
        best_model_filename = None  # Ensure this is always defined
        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                hidden = self.model.init_hidden(batch_X.size(0), self.device)
                optimizer.zero_grad()
                output, _ = self.model(batch_X, hidden)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

            # Save the model at each epoch
            model_filename = os.path.join(save_dir, f"music_model_epoch{epoch+1}_loss{avg_loss:.4f}.pth")
            torch.save(self.model.state_dict(), model_filename)
            print(f"Model saved to {model_filename}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_filename = os.path.join(save_dir, f"best_music_model.pth")
                torch.save(self.model.state_dict(), best_model_filename)
                print(f"Best model saved to {best_model_filename}")
        
        if best_model_filename:
            print(f"Training complete. Best model saved to {best_model_filename}")
        else:
            print(f"Training complete. No best model was saved.")

##CONSTANTS##
#quatre mesures
#RELATED TO DATA
SEQLENGTH =64
MIDI_FOLDER = 'data/'
SAMPLING_RATE=4

###TO TRAIN###
NUM_EPOCHS=30
BATCH_SIZE=64
SAVE_DIRECTORY='models/'
HIDDEN_SIZE =300
NUM_LAYERS=3
DROPOUT =0.2
LEARNING_RATE=0.001

# Main training script
def train_or_resume(midi_folder: str, save_dir: str, num_epochs: int = 1000, batch_size: int = 64, 
                    learning_rate: float = 0.001, sequence_length: int = 100, sampling_rate=4,hidden_size=256, num_layers=2,dropout=0.2):
    processor = PianoRollProcessor(fs=sampling_rate)
    trainer = MusicTrainer(processor, sequence_length, hidden_size, num_layers, dropout)
    
    # Find the latest model in the save directory to resume from
    model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
    latest_model = sorted(model_files, key=lambda x: int(x.split('_epoch')[1].split('_')[0]), reverse=True)
    
    if latest_model:
        model_path = os.path.join(save_dir, latest_model[0])
        print(f"Resuming training from {model_path}")
        trainer.model.load_state_dict(torch.load(model_path))
        epoch_num = int(model_path.split('_epoch')[-1].split('_')[0])
        print(f"Resuming from epoch {epoch_num + 1}")
    else:
        print(f"Training from scratch.")
        epoch_num = 0
    
    trainer.train(midi_folder, num_epochs, batch_size, learning_rate, save_dir, start_epoch=epoch_num)

if __name__ == "__main__":
    train_or_resume(MIDI_FOLDER, SAVE_DIRECTORY, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, 
                    learning_rate=LEARNING_RATE, sequence_length=SEQLENGTH,
                    sampling_rate=SAMPLING_RATE,hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,dropout=DROPOUT)
