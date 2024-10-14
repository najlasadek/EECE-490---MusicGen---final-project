from music21 import *
import matplotlib.pyplot as plt
import numpy as np
import glob
from music21 import *
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

def process_data(songs):
    elements_count = 0
    whole_data = []
    for song in songs:
        midi_data = converter.parse(song).flat
        song_data = []
        prev_offset = -1
        for element in midi_data:
            elements_count+=1
            if isinstance(element, note.Note):
                if element.offset != prev_offset:
                    song_data.append([element.pitch.nameWithOctave, 
                                      element.quarterLength])
                else:
                    if len(song_data[-1]) < 4:
                        song_data[-1].append(element.pitch.nameWithOctave)   
                        song_data[-1].append(element.quarterLength)       
                prev_offset = element.offset
            elif isinstance(element, chord.Chord):
                pitch_names = '.'.join(n.nameWithOctave for n in element.pitches)
                if element.offset != prev_offset:
                    song_data.append([pitch_names, element.quarterLength])
                else:
                    if len(song_data[-1]) < 4:
                        song_data[-1].append(pitch_names)   
                        song_data[-1].append(element.quarterLength)      
                prev_offset = element.offset
        for item in song_data:
            if len(item) < 4:
                item.append(None)
                item.append(None)
        print("Current_Song_data",song_data)
        for item_4 in song_data:
         whole_data.append(item_4)
    print("Elements count",elements_count)
    return whole_data

#because tuples are hashable
def transform_data(songs):
    transform_data = []
    for items_4 in songs:
        transform_data.append(tuple(items_4))
    return transform_data
    

####STEP1####
#ENCODE THE DATA AND SAVE IT
DATA="midi_folder"
#create a folder result
print(f"Will process MIDI files in the f'{DATA}' folder")
pieces= glob.glob(f'{DATA}/*.mid')
parsed=process_data(pieces)
print("Parsed",parsed)
encoded=transform_data(parsed)
print("Encoded",encoded)

print("\nSaving the encoding of musical elements to file...")
with open('result/encodingsimple', 'wb') as filepath:
    pickle.dump(encoded, filepath)
print("Saved successfully as encoding!")

print("\nSaving the parsed musical elements to file...")
with open('result/parsedsimple', 'wb') as filepath:
    pickle.dump(parsed, filepath)
print("Saved successfully as parsed!")

#TRANSFORM EACH COMBINATION INTO A NUMBER
unique_encodings =set(item for item in encoded)
music_to_int = dict((note, number) for number, note in enumerate(unique_encodings))
print(f"\nUnique musical structures: {len(music_to_int)}")
print("First 25 mappings in music_to_int dictionary:")
for i, (note, number) in enumerate(list(music_to_int.items())[:25]):
    print(f"{note}: {number}")

####PREPARE DATA FOR THE LSTM####
sequence_length = 50
network_input = []
network_output = []
n_vocab = len(unique_encodings)
print(f"\nVocabulary size/Number of unique encodings: {n_vocab}")
print("\nCreating input sequences and corresponding outputs...")
print(f"Sequence length for training: {sequence_length}")
for i in range(0, len(encoded) - sequence_length, 1):
    sequence_in = encoded[i:i + sequence_length]
    #print("Sequence_in",sequence_in)
    sequence_out = encoded[i + sequence_length]
    #print("Sequence_out",sequence_out)
    network_input.append([music_to_int[char] for char in sequence_in])
    network_output.append(music_to_int[sequence_out])
#print("Network_input",network_input)
#print("Network_output",network_output)
print(f"Total input sequences created: {len(network_input)}")
print(f"Example input sequence: {network_input[0]}...")
print(f"Example output: {network_output[0]}")



print("\nReshaping the input into a format compatible with LSTM layers")
n_patterns = len(network_input)
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
print(f"Reshaped input shape (#of sequences ,sequence_length,#of features): {network_input.shape}")

print("Normalizing input")
network_input = network_input / float(n_vocab)
print(f"Sample normalized input: {network_input[0][0]}")

print("\nConverting network output to categorical")
network_output = to_categorical(network_output)
print(f"Converted output shape to categorical (#sequences, #unique_encodings) {network_output.shape}")

####BUILD MODEL LSTM####
print("\nCreating neural network model...")
model = Sequential([
    Input(shape=(sequence_length, 1)),
    LSTM(512, return_sequences=True),
    Dropout(0.3),
    LSTM(512, return_sequences=True),
    Dropout(0.3),
    LSTM(512),
    Dense(256),
    Dropout(0.3),
    Dense(n_vocab),
    Activation('softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print("Model created successfully!")
model.summary()


####TRAIN####
print("\nStarting model training...")
filepath = "weightssimple-{epoch:02d}-{loss:.4f}-bigger.keras"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint]
history = model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

print("\nTraining completed!")
print(f"Final loss: {history.history['loss'][-1]}")











