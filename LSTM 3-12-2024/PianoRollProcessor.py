import pretty_midi
import numpy as np
from typing import Tuple
import os

#fs is a hyperparameter - frequency
class PianoRollProcessor:
    def __init__(self, lowest_note: int = 21, highest_note: int = 108, fs: int = 20):
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
        return X, y


# Test Case
if __name__ == "__main__":
    # Define test paths
    input_midi_file = "Prelude2.mid"  # Path to the input MIDI file
    encoded_output_piano_roll = "test_piano_roll.npy"  # Temp file for encoded piano roll
    decoded_output_midi_file = "test_output.mid"  # Path to the reconstructed MIDI file

    # Create a test processor instance
    processor = PianoRollProcessor()

    # Test case: Encode MIDI to piano roll
    print("\n--- Test Case: Encoding MIDI to Piano Roll ---")
    try:
        piano_roll, duration = processor.midi_to_piano_roll(input_midi_file)
        print("Piano roll encoding successful!")
        np.save(encoded_output_piano_roll, piano_roll)  # Save for verification
    except Exception as e:
        print(f"Error during piano roll encoding: {e}")

    # Test case: Decode piano roll back to MIDI
    print("\n--- Test Case: Decoding Piano Roll to MIDI ---")
    try:
        # Load the piano roll from the saved file (simulates intermediate processing)
        loaded_piano_roll = np.load(encoded_output_piano_roll)
        processor.piano_roll_to_midi(loaded_piano_roll, decoded_output_midi_file)
        print("MIDI reconstruction successful!")
    except Exception as e:
        print(f"Error during MIDI reconstruction: {e}")

    # Clean up temporary files
    if os.path.exists(encoded_output_piano_roll):
        os.remove(encoded_output_piano_roll)
    print("\n--- Test Case Completed ---")