import mido

def unify_tempo(input_midi, output_midi, unified_tempo=1000000):
    """
    Remove all tempo marks from a MIDI file and set a unified tempo.
    
    Parameters:
        input_midi (str): Path to the input MIDI file.
        output_midi (str): Path to save the output MIDI file.
        unified_tempo (int): New tempo in microseconds per beat (default is 1,000,000 for 60 BPM).
    """
    # Load the MIDI file
    mid = mido.MidiFile(input_midi)

    print(f"Original Ticks Per Beat: {mid.ticks_per_beat}")
    print(f"Setting unified tempo to {unified_tempo} (60 BPM).")

    new_tracks = []

    # Process each track
    for i, track in enumerate(mid.tracks):
        new_track = mido.MidiTrack()
        for msg in track:
            # Skip tempo messages
            if not (msg.type == 'set_tempo'):
                new_track.append(msg)
        # Add the unified tempo at the beginning of the first track
        if i == 0:
            new_track.insert(0, mido.MetaMessage('set_tempo', tempo=unified_tempo))
        new_tracks.append(new_track)

    # Create a new MIDI file and save it
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    new_mid.tracks.extend(new_tracks)
    new_mid.save(output_midi)

    print(f"Saved new MIDI file with unified tempo to: {output_midi}")

# Example usage
input_file = "Fugue18.mid"  # Replace with your MIDI file
output_file = "Fugue18normalized.mid"  # Replace with the desired output file name
unify_tempo(input_file, output_file, unified_tempo=1000000)  # Set tempo to 1,000,000 µs per beat (60 BPM)
