from GenerationTools import generate_piece

## CONSTANTS ##
MODEL_PATH = "models/music_model_epoch70_loss0.0815.pth"    # Path to your trained model
MIDI_FOLDER = "data/"             # Folder containing MIDI files for seeding
OUTPUT_PATH = "generated_piece_2.mid"
MINUTES = 1.0                     # Length of piece in minutes
TEMPERATURE = 0.8                 # Higher = more random
SEQLENGTH = 20                    # Sequence length for LSTM

def main():
    print("\n=== Starting Music Generation Pipeline ===")
    print(f"Model path: {MODEL_PATH}")
    print(f"MIDI folder: {MIDI_FOLDER}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Length: {MINUTES} minutes")
    print(f"Temperature: {TEMPERATURE}")

    # Generate a single piece of music
    generate_piece(
        model_path=MODEL_PATH,
        midi_folder=MIDI_FOLDER,
        output_path=OUTPUT_PATH,
        minutes=MINUTES,
        temperature=TEMPERATURE,
        sequence_length=SEQLENGTH
    )
    print("\nGeneration complete!")

if __name__ == "__main__":
    main()
