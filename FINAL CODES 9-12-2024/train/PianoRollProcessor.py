import pretty_midi
import numpy as np
from typing import Tuple
import os

# fs is a hyperparameter - frequency
class PianoRollProcessor:
    def __init__(self, lowest_note: int = 21, highest_note: int = 108, fs: int = 4):
        self.lowest_note = lowest_note
        self.highest_note = highest_note
        self.note_range = highest_note - lowest_note + 1  # Should be 88
        self.fs = fs
        print(f"Initialized processor with note range: {self.note_range} notes ({lowest_note}-{highest_note})")
        print(f"Sampling rate: {fs}Hz ({fs} steps per second)")
        assert self.note_range == 88, "Error: The note range is not 88. Check 'lowest_note' and 'highest_note' settings."

    def midi_to_piano_roll(self, midi_path: str) -> Tuple[np.ndarray, float]:
        """
        Converts a MIDI file to a piano roll representation.
        
        Parameters:
            midi_path (str): Path to the input MIDI file.
            
        Returns:
            Tuple[np.ndarray, float]: The piano roll matrix and the duration of the MIDI file.
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

        # Debugging statement to confirm dimensions
        assert piano_roll.shape[1] == 88, f"Error: Piano roll has {piano_roll.shape[1]} notes instead of 88."
        print(f"Final piano roll dimensions: {piano_roll.shape} (timesteps x 88)")
        print(f"Duration: {duration:.2f}s, Total steps: {total_steps}, Total notes: {note_count}")
        return piano_roll, duration

    def piano_roll_to_midi(self, piano_roll: np.ndarray, output_path: str, velocity: int = 100) -> None:
        """
        Converts a piano roll back into a MIDI file.
        
        Parameters:
            piano_roll (np.ndarray): The piano roll matrix to convert.
            output_path (str): Path to save the output MIDI file.
            velocity (int): The velocity of the MIDI notes.
        """
        print(f"\nConverting piano roll to MIDI. Input dimensions: {piano_roll.shape}")
        assert piano_roll.shape[1] == 88, f"Error: Piano roll width is not 88. It is {piano_roll.shape[1]}."

        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        note_count = 0

        for note_num in range(self.note_range):
            active = np.where(piano_roll[:, note_num] > 0)[0]
            
            if len(active) > 0:
                splits = np.where(np.diff(active) > 1)[0] + 1
                segments = np.split(active, splits)
                
                for segment in segments:
                    if len(segment) > 0:
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=note_num + self.lowest_note,
                            start=segment[0] / self.fs,
                            end=(segment[-1] + 1) / self.fs
                        )
                        instrument.notes.append(note)
                        note_count += 1

        pm.instruments.append(instrument)
        pm.write(output_path)
        print(f"Saved MIDI file with {note_count} notes to {output_path}")

    def prepare_sequences(self, piano_roll: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares sequences for LSTM training from the piano roll.
        
        Parameters:
            piano_roll (np.ndarray): The piano roll matrix (timesteps x notes).
            sequence_length (int): The length of input sequences for the LSTM.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Input sequences (X) and their corresponding targets (y).
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
        print(f"Sample X sequence: {X[0]}")  # Sample input sequence
        print(f"Sample y sequence: {y[0]}")  # Sample target sequence
        return X, y


def main():
    # Initialize the processor
    processor = PianoRollProcessor(lowest_note=21, highest_note=108, fs=4)

    # Input MIDI file path
    input_midi = "gooddata.mid"  # Replace with the path to your MIDI file
    output_midi = "example_output.mid"  # Path to save the reconstructed MIDI file

    # Step 1: Encode MIDI to piano roll
    print("\nStep 1: Converting MIDI to piano roll...")
    piano_roll, duration = processor.midi_to_piano_roll(input_midi)
    print(f"Piano roll created. Shape: {piano_roll.shape}, Duration: {duration:.2f} seconds")

    # Step 2: Convert piano roll back to MIDI
    print("\nStep 2: Converting piano roll back to MIDI...")
    processor.piano_roll_to_midi(piano_roll, output_midi)
    print(f"Reconstructed MIDI saved to {output_midi}")

if __name__ == "__main__":
    main()

'''a variation of prepare sequences to split the dara
'''
'''
    def prepare_sequences(self, piano_roll: np.ndarray, sequence_length: int):
        """
        Prepares sequences for LSTM training from the piano roll, keeping only the first half.
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

        # Keep only the first half of the sequences
        total_sequences = len(X)
        half_point = total_sequences // 1  # Calculate the midpoint
        print(f"Total sequences before slicing: {total_sequences}")

        X = X[:half_point]  # Keep only the first half of the sequences
        y = y[:half_point]  # Keep only the first half of the targets

        print(f"Total sequences after slicing: {len(X)}")
        return X, y
'''