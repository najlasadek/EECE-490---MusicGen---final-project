from PianoRollProcessor import PianoRollProcessor
from MusicTrainer import MusicTrainer
from tensorflow.keras.models import load_model


##CONSTANTS##
#quatre mesures
#RELATED TO DATA
SEQLENGTH =64
MIDI_FOLDER = 'data/'
SAMPLING_RATE=4

###TO TRAIN###
NUM_EPOCHS=30
BATCH_SIZE=64
SAVE_PATH= 'music_model.pth'
HIDDEN_SIZE =300
NUM_LAYERS=3
DROPOUT =0.2
LEARNING_RATE=0.001
'''
####TO GENERATE
MODEL_PATH = "music_model.pth"    # Path to your trained model
MIDI_FOLDER = "data/"             # Folder containing MIDI files for seeding
OUTPUT_PATH = "generated_piece.mid"
MINUTES = 2.0                     # Length of piece in minutes
TEMPERATURE = 0.8                 # Higher = more random
'''



print("\n=== Starting Music Generation Pipeline ===")
print("\nInitializing components...")
processor_piano = PianoRollProcessor(fs=SAMPLING_RATE)


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
        save_path= SAVE_PATH,
        learning_rate=LEARNING_RATE
    )


'''
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

'''
