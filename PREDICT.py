import pickle
import numpy
from music21 import instrument, note, stream, chord,midi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, Input

ENCODED='result/encodingsimple'
PARSED='result/parsedsimple'


def prepare_sequences(encoded, unique_encodings, n_vocab):
    # map between notes and integers and back
    music_to_int = dict((note, number) for number, note in enumerate(unique_encodings))
    sequence_length = 100
    network_input = []
    # output = []
    for i in range(0, len(encoded) - sequence_length, 1):
        sequence_in = encoded[i:i + sequence_length]
        # sequence_out = notes[i + sequence_length]
        network_input.append([music_to_int[char] for char in sequence_in])
        # output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)



    return (network_input, normalized_input)


def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights to each node
    model.load_weights('weightssimple-195-0.2771-bigger.keras')

    return model


def generate_notes(model, network_input, unique_encodings, n_vocab):

    int_to_music= dict((number, note) for number, note in enumerate(unique_encodings))
    print("The int to music decoding function",int_to_music)

    # pick a random sequence from the input as a starting point for the prediction
    start = numpy.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        #print(prediction)

        index = numpy.argmax(prediction)
        print(index)
        result = int_to_music[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def create_midi(prediction_output,i):
    # write the generated songs as midi files

        p1 = stream.Part()
        p1.insert(0, instrument.Piano())
        p2 = stream.Part()
        p2.insert(0, instrument.Piano())
        for item_4 in prediction_output:
            if item_4 != (None, None, None, None):
                if item_4[0] != None and item_4[1] != None:
                    if '.' in item_4[0]:
                        chord_pitches = item_4[0].split('.')
                        p1.append(chord.Chord(chord_pitches, quarterLength = item_4[1]))
                    else:
                        p1.append(note.Note(item_4[0], quarterLength = item_4[1]))
                if item_4[2] != None and item_4[3] != None:
                    if '.' in item_4[2]:
                        chord_pitches = item_4[2].split('.')
                        p2.append(chord.Chord(chord_pitches, quarterLength = item_4[3]))
                    else:
                        p2.append(note.Note(item_4[2], quarterLength = item_4[3]))
        s = stream.Stream([p1, p2])
        mid = midi.translate.streamToMidiFile(s)
        mid.open('outsimple'+str(i)+'.mid', 'wb')
        mid.write()
        mid.close()
        print("saved generated song! check your directory.")





####MAIN####
def generate():
    # load the notes used to train the model
    print("Loading the notes used to train the model parsed and encoded")
    with open(ENCODED, 'rb') as filepath:
        encoded = pickle.load(filepath)
        print(encoded)
    with open(PARSED, 'rb') as filepath:
        parsed = pickle.load(filepath)
      
    # Get all pitch names
    unique_encodings =set(item for item in encoded)
    # Get all pitch names
    n_vocab = len(unique_encodings)

    print("Getting network input")
    
    network_input, normalized_input = prepare_sequences(encoded, unique_encodings, n_vocab)
    print("Creating model")
    model = create_network(normalized_input, n_vocab)
    print("Generating notes")
    for i in range (1,11,1):
        prediction_output = generate_notes(model, network_input, unique_encodings, n_vocab)
        print("Creating midi file",i)
        create_midi(prediction_output,i)
    
if __name__ == '__main__':
    generate()