import pretty_midi
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class PianoRollProcessor:
    def __init__(self, lowest_note: int = 21, highest_note: int = 108, fs: int = 30):
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

        # Generate a mesmerizing plot of the piano roll
        self.plot_piano_roll(piano_roll, duration)

        return piano_roll, duration

    def plot_piano_roll(self, piano_roll: np.ndarray, duration: float):
        """
        Plots the piano roll matrix as a heatmap for visualization with a black background,
        golden notes, and horizontal lines for all 88 keys.
        """
        plt.figure(figsize=(12, 6))
        plt.imshow(piano_roll.T, aspect='auto', origin='lower', cmap='inferno')  # Inferno provides golden-like tones
        
        # Add horizontal lines for all 88 piano keys
        for i in range(88):
            plt.hlines(i - 0.5, 0, piano_roll.shape[0], color='gray', alpha=0.2, linewidth=0.5)

        plt.title("Piano Roll Visualization", color="white")
        plt.xlabel("Time Steps", color="white")
        plt.ylabel("Piano Keys (0 to 88)", color="white")

        # Format x and y ticks with white color
        plt.xticks(
            ticks=np.linspace(0, piano_roll.shape[0], num=10),
            labels=[f"{t:.1f}s" for t in np.linspace(0, duration, num=10)],
            color="white"
        )
        plt.yticks(
            ticks=np.linspace(0, 87, num=9),  # Divide 0 to 88 into equal intervals
            labels=[f"{int(k)}" for k in np.linspace(0, 88, num=9)],  # Label keys from 0 to 88
            color="white"
        )

        # Black background
        plt.gca().set_facecolor("black")
        plt.gcf().patch.set_facecolor("black")

        # Tight layout and show the plot
        plt.tight_layout()
        plt.show()




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
    

if __name__ == "__main__":
    # Define the test MIDI file path
    input_midi_file = "Fugue18.mid"  # Replace with the path to your MIDI file

    # Create a PianoRollProcessor instance
    processor = PianoRollProcessor()

    # Test case: MIDI to Piano Roll and Visualization
    print("\n--- Test Case: MIDI to Piano Roll and Visualization ---")
    try:
        # Convert MIDI to piano roll and visualize
        piano_roll, duration = processor.midi_to_piano_roll(input_midi_file)

        # Save the piano roll matrix as a .npy file (optional)
        np.save("test_piano_roll.npy", piano_roll)
        print("Piano roll matrix saved successfully!")
    except Exception as e:
        print(f"Error during MIDI to piano roll conversion: {e}")

    # Note: Further tests can include loading the saved piano roll and passing it to
    # the LSTM model for training or generating new MIDI files from the piano roll.


