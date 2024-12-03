from PianoRollProcessor import PianoRollProcessor
from MusicTrainer import MusicTrainer
from tensorflow.keras.models import load_model
from GenerationTools import generate_piece

##CONSTANTS##
SEQLENGTH =20
MIDI_FOLDER = 'data/'

###TO TRAIN###
NUM_EPOCHS=1000
BATCH_SIZE=64
SAVE_PATH= 'music_model.pth'
HIDDEN_SIZE =256
NUM_LAYERS=2
DROPOUT =0.2


####TO GENERATE
MODEL_PATH = "music_model.pth"    # Path to your trained model
MIDI_FOLDER = "data/"             # Folder containing MIDI files for seeding
OUTPUT_PATH = "generated_piece.mid"
MINUTES = 2.0                     # Length of piece in minutes
TEMPERATURE = 0.8                 # Higher = more random



print("\n=== Starting Music Generation Pipeline ===")
print("\nInitializing components...")
processor_piano = PianoRollProcessor()


model = MusicTrainer(
        processor = processor_piano,
        sequence_length= SEQLENGTH, 
        hidden_size =HIDDEN_SIZE, 
        num_layers = NUM_LAYERS, 
        dropout= DROPOUT
        ) 

print("\nStarting training...")
model.train(
        midi_folder=MIDI_FOLDER,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        save_path= SAVE_PATH
    )


print("\n=== Music Generation Script ===")
print(f"Model path: {MODEL_PATH}")
print(f"MIDI folder: {MIDI_FOLDER}")
print(f"Output: {OUTPUT_PATH}")
print(f"Length: {MINUTES} minutes")
print(f"Temperature: {TEMPERATURE}")

# Generate a single piece
generate_piece(
    model_path=MODEL_PATH,
    midi_folder=MIDI_FOLDER,
    output_path=OUTPUT_PATH,
    minutes=MINUTES,
    temperature=TEMPERATURE,
    sequence_length = SEQLENGTH
)
