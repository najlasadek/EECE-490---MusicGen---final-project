import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, List
import pretty_midi
import os
import random
from PianoRollProcessor import PianoRollProcessor
from MusicTrainer import MusicLSTM

def load_random_midi_seed(midi_folder: str, processor: PianoRollProcessor, sequence_length: int) -> np.ndarray:
    """Load a random segment from a random MIDI file to use as seed"""
    print("\nLoading seed from MIDI files...")

    midi_files = [f for f in os.listdir(midi_folder) if f.endswith('.mid')]
    if not midi_files:
        raise ValueError(f"No MIDI files found in {midi_folder}")
    
    midi_file = np.random.choice(midi_files)
    print(f"Selected seed file: {midi_file}")
    
    piano_roll, _ = processor.midi_to_piano_roll(os.path.join(midi_folder, midi_file))
    
    if len(piano_roll) < sequence_length:
        raise ValueError(f"MIDI file {midi_file} is too short for sequence length {sequence_length}")
    
    # Try to find a section with some notes
    max_attempts = 10
    for attempt in range(max_attempts):
        start_idx = np.random.randint(0, len(piano_roll) - sequence_length)
        seed_sequence = piano_roll[start_idx:start_idx + sequence_length]
        if np.sum(seed_sequence) > 0:  # Check if there are any notes
            break
    
    print(f"Extracted seed sequence from position {start_idx}")
    active_notes = np.sum(seed_sequence > 0)
    print(f"Seed contains {active_notes} active notes")
    
    return seed_sequence


def generate_long_music(model_path: str,
                       midi_folder: str,
                       minutes: float = 1.0,
                       sequence_length: int = 0,
                       temperature: float = 0.8,
                       batch_size: int = 1000) -> np.ndarray:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    processor = PianoRollProcessor()
    model = MusicLSTM(
        input_size=processor.note_range,
        hidden_size=512,
        num_layers=4
    ).to(device)
    
    steps_per_minute = 60 * processor.fs
    total_steps = int(minutes * steps_per_minute)
    
    print(f"\nGenerating {minutes:.1f} minutes of music")
    print(f"Total steps needed: {total_steps} ({total_steps/processor.fs:.1f} seconds)")
    
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    sequence = load_random_midi_seed(midi_folder, processor, sequence_length)
    generated = list(sequence)
    
    remaining_steps = total_steps
    batch_count = 0
    
    with torch.no_grad():
        while remaining_steps > 0:
            current_batch = min(batch_size, remaining_steps)
            batch_count += 1
            print(f"\nGenerating batch {batch_count}, {current_batch} steps")
            
            hidden = model.init_hidden(1, device)
            
            for i in range(current_batch):
                if i % 100 == 0:
                    seconds_done = len(generated) / processor.fs
                    minutes_done = int(seconds_done // 60)
                    seconds_part = seconds_done % 60
                    print(f"Generated {minutes_done}m {seconds_part:.1f}s", end='\r')
                
                x = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                output, hidden = model(x, hidden)
                
                output = output.squeeze(0)
                if temperature != 1.0:
                    output = torch.log(output) / temperature
                    output = torch.exp(output)
                    output = output / torch.sum(output)
                
                probs = output.cpu().numpy()
                notes = (np.random.random(probs.shape) < probs).astype(float)
                
                generated.append(notes)
                sequence = np.array(generated[-sequence_length:])
            
            remaining_steps -= current_batch
            print(f"Completed batch {batch_count}, {remaining_steps} steps remaining")
    
    return np.array(generated)

def generate_piece(model_path: str,
                  midi_folder: str,
                  output_path: str = "generated_piece.mid",
                  minutes: float = 1.0,
                  temperature: float = 0.8,
                  sequence_length : int = 100):
    """Generate a single piece of music"""
    print("\n=== Generating New Piece ===")
    print(f"Length: {minutes:.1f} minutes")
    print(f"Temperature: {temperature}")
    
    generated = generate_long_music(
        model_path=model_path,
        midi_folder=midi_folder,
        minutes=minutes,
        temperature=temperature,
        sequence_length= sequence_length
    )
    
    processor = PianoRollProcessor()
    processor.piano_roll_to_midi(generated, output_path)
    print("\nGeneration complete!")
