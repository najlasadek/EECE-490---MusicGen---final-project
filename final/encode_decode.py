import pretty_midi
import numpy as np
from typing import Tuple
import os

class PianoRollProcessor:
    def __init__(self, lowest_note: int = 21, highest_note: int = 109, fs: int = 100):
        self.lowest_note = lowest_note
        self.highest_note = highest_note
        self.note_range = highest_note - lowest_note + 1
        self.fs = fs
        print(f"Initialized processor with note range: {self.note_range} notes ({lowest_note}-{highest_note})")
        print(f"Sampling rate: {fs}Hz ({fs} steps per second)")
    
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
                        
        print(f"Duration: {duration:.2f}s, Total steps: {total_steps}, Notes: {note_count}")
        return piano_roll, duration
    
    def piano_roll_to_midi(self, piano_roll: np.ndarray, output_path: str, velocity: int = 100) -> None:
        """
        Converts a piano roll back into a MIDI file.
        
        Parameters:
            piano_roll (np.ndarray): The piano roll matrix to convert.
            output_path (str): Path to save the output MIDI file.
            velocity (int): The velocity of the MIDI notes.
        """
        duration_seconds = len(piano_roll) / self.fs
        minutes = int(duration_seconds // 60)
        seconds = duration_seconds % 60
        print(f"\nConverting piano roll to MIDI: {minutes}m {seconds:.1f}s of music")
        
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


# Test Case
if __name__ == "__main__":
    # Define test paths
    input_midi_file = "wohl_1.mid"  # Path to the input MIDI file
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
