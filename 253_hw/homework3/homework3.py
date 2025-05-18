
# import required packages
import random
from glob import glob
from collections import defaultdict

import numpy as np
from numpy.random import choice

from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile

duration2length = {
    '0.2.8': 2,  # sixteenth note, 0.25 beat in 4/4 time signature
    '0.4.8': 4,  # eighth note, 0.5 beat in 4/4 time signature
    '1.0.8': 8,  # quarter note, 1 beat in 4/4 time signature
    '2.0.8': 16, # half note, 2 beats in 4/4 time signature
    '4.0.4': 32, # whole note, 4 beats in 4/4 time signature
}

# In[4]:


# You can change the random seed but try to keep your results deterministic!
# If I need to make changes to the autograder it'll require rerunning your code,
# so it should ideally generate the same results each time.
random.seed(42)


# ### Load music dataset
# We will use a subset of the [PDMX dataset](https://zenodo.org/records/14984509).
# 
# Please find the link in the homework spec.
# 
# All pieces are monophonic music (i.e. one melody line) in 4/4 time signature.

# In[7]:


midi_files = glob('PDMX_subset/*.mid')
len(midi_files)


# ### Train a tokenizer with the REMI method in MidiTok

# In[8]:


config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=midi_files)


def note_extraction(midi_file):
    """
    Extract note pitch events from a midi file
    
    Args:
        midi_file: Path to a midi file
        
    Returns:
        A list of note pitch events (e.g. [60, 62, 61, ...])
    """
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens
    
    notes = []
    for i in range(len(tokens)):
        if tokens[i].startswith('Pitch_'):
            # Extract the pitch value from the token (format: 'Pitch_XX')
            pitch = int(tokens[i].split('_')[1])
            notes.append(pitch)
    
    return notes

def note_frequency(midi_files):
    """
    Count frequency of each note pitch in all midi files
    
    Args:
        midi_files: List of paths to midi files
        
    Returns:
        A dictionary mapping note pitch events to their frequency
    """
    note_counts = {}
    
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        for note in notes:
            if note in note_counts:
                note_counts[note] += 1
            else:
                note_counts[note] = 1
    
    return note_counts

def note_unigram_probability(midi_files):
    """
    Calculate unigram probability for each note
    
    Args:
        midi_files: List of paths to midi files
        
    Returns:
        A dictionary mapping note pitch events to probabilities
    """
    note_counts = note_frequency(midi_files)
    unigramProbabilities = {}
    
    # Calculate total count of all notes
    total_notes = sum(note_counts.values())
    
    # Normalize counts to get probabilities
    for note, count in note_counts.items():
        unigramProbabilities[note] = count / total_notes
    
    return unigramProbabilities

def note_bigram_probability(midi_files):
    """
    Calculate bigram probabilities p(next_note | previous_note)
    
    Args:
        midi_files: List of paths to midi files
        
    Returns:
        Two dictionaries:
        - bigramTransitions: maps each note to list of next notes
        - bigramTransitionProbabilities: maps each note to list of next note probabilities
    """
    bigramTransitions = defaultdict(list)
    bigramTransitionProbabilities = defaultdict(list)
    
    # Count bigram occurrences
    bigram_counts = defaultdict(lambda: defaultdict(int))
    
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        
        # Count transitions between consecutive notes
        for i in range(len(notes) - 1):
            prev_note = notes[i]
            next_note = notes[i + 1]
            bigram_counts[prev_note][next_note] += 1
    
    # Convert counts to probabilities
    for prev_note, next_notes in bigram_counts.items():
        total = sum(next_notes.values())
        
        # Create ordered lists of next notes and their probabilities
        for next_note, count in next_notes.items():
            bigramTransitions[prev_note].append(next_note)
            bigramTransitionProbabilities[prev_note].append(count / total)
    
    return bigramTransitions, bigramTransitionProbabilities

def sample_next_note(note):
    """
    Sample the next note based on bigram probabilities
    
    Args:
        note: Previous note
        
    Returns:
        Next note sampled from pairwise probabilities
    """
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    
    if note not in bigramTransitions:
        # If we haven't seen this note before, pick a random note from the unigram distribution
        unigramProbabilities = note_unigram_probability(midi_files)
        notes = list(unigramProbabilities.keys())
        probabilities = list(unigramProbabilities.values())
        return np.random.choice(notes, p=probabilities)
    
    # Sample from the bigram distribution
    next_notes = bigramTransitions[note]
    probabilities = bigramTransitionProbabilities[note]
    
    return np.random.choice(next_notes, p=probabilities)

def note_bigram_perplexity(midi_file):
    """
    Calculate perplexity of the bigram model on a midi file
    
    Args:
        midi_file: Path to a midi file
        
    Returns:
        Perplexity value
    """
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    
    notes = note_extraction(midi_file)
    
    if len(notes) <= 1:
        return float('inf')  # Handle edge case of very short sequences
    
    # Calculate log probabilities sum
    log_prob_sum = 0
    
    # Handle first note using unigram probability
    if notes[0] in unigramProbabilities:
        first_note_prob = unigramProbabilities[notes[0]]
    else:
        # Smoothing for unseen notes
        first_note_prob = 1e-10
    log_prob_sum += np.log(first_note_prob)
    
    # Handle remaining notes using bigram probabilities
    for i in range(1, len(notes)):
        prev_note = notes[i-1]
        curr_note = notes[i]
        
        if prev_note in bigramTransitions:
            if curr_note in bigramTransitions[prev_note]:
                # Get the index of the current note in the transitions list
                idx = bigramTransitions[prev_note].index(curr_note)
                prob = bigramTransitionProbabilities[prev_note][idx]
            else:
                # Smoothing for unseen transitions
                prob = 1e-10
        else:
            # If we've never seen the previous note, use unigram probability
            prob = unigramProbabilities.get(curr_note, 1e-10)
        
        log_prob_sum += np.log(prob)
    
    # Calculate perplexity
    avg_log_prob = log_prob_sum / len(notes)
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity

def note_trigram_probability(midi_files):
    """
    Calculate trigram probabilities p(next_note | next_previous_note, previous_note)
    
    Args:
        midi_files: List of paths to midi files
        
    Returns:
        Two dictionaries:
        - trigramTransitions: maps (next_previous_note, previous_note) to list of next notes
        - trigramTransitionProbabilities: maps (next_previous_note, previous_note) to list of next note probabilities
    """
    trigramTransitions = defaultdict(list)
    trigramTransitionProbabilities = defaultdict(list)
    
    # Count trigram occurrences
    trigram_counts = defaultdict(lambda: defaultdict(int))
    
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        
        # Count transitions between three consecutive notes
        for i in range(len(notes) - 2):
            next_prev_note = notes[i]
            prev_note = notes[i + 1]
            next_note = notes[i + 2]
            
            trigram_counts[(next_prev_note, prev_note)][next_note] += 1
    
    # Convert counts to probabilities
    for (next_prev_note, prev_note), next_notes in trigram_counts.items():
        total = sum(next_notes.values())
        
        # Create ordered lists of next notes and their probabilities
        for next_note, count in next_notes.items():
            trigramTransitions[(next_prev_note, prev_note)].append(next_note)
            trigramTransitionProbabilities[(next_prev_note, prev_note)].append(count / total)
    
    return trigramTransitions, trigramTransitionProbabilities

def note_trigram_perplexity(midi_file):
    """
    Calculate perplexity of the trigram model on a midi file
    
    Args:
        midi_file: Path to a midi file
        
    Returns:
        Perplexity value
    """
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    
    notes = note_extraction(midi_file)
    
    if len(notes) <= 2:
        return float('inf')  # Handle edge case of very short sequences
    
    # Calculate log probabilities sum
    log_prob_sum = 0
    
    # Handle first note using unigram probability
    first_note_prob = unigramProbabilities.get(notes[0], 1e-10)
    log_prob_sum += np.log(first_note_prob)
    
    # Handle second note using bigram probability
    if notes[0] in bigramTransitions:
        if notes[1] in bigramTransitions[notes[0]]:
            idx = bigramTransitions[notes[0]].index(notes[1])
            second_note_prob = bigramTransitionProbabilities[notes[0]][idx]
        else:
            # Smoothing for unseen transitions
            second_note_prob = 1e-10
    else:
        second_note_prob = unigramProbabilities.get(notes[1], 1e-10)
    
    log_prob_sum += np.log(second_note_prob)
    
    # Handle remaining notes using trigram probabilities
    for i in range(2, len(notes)):
        next_prev_note = notes[i-2]
        prev_note = notes[i-1]
        curr_note = notes[i]
        
        if (next_prev_note, prev_note) in trigramTransitions:
            if curr_note in trigramTransitions[(next_prev_note, prev_note)]:
                idx = trigramTransitions[(next_prev_note, prev_note)].index(curr_note)
                prob = trigramTransitionProbabilities[(next_prev_note, prev_note)][idx]
            else:
                # Smoothing for unseen trigram transitions
                prob = 1e-10
        else:
            # Fall back to bigram if trigram not found
            if prev_note in bigramTransitions:
                if curr_note in bigramTransitions[prev_note]:
                    idx = bigramTransitions[prev_note].index(curr_note)
                    prob = bigramTransitionProbabilities[prev_note][idx]
                else:
                    prob = 1e-10
            else:
                # Fall back to unigram if bigram not found
                prob = unigramProbabilities.get(curr_note, 1e-10)
        
        log_prob_sum += np.log(prob)
    
    # Calculate perplexity
    avg_log_prob = log_prob_sum / len(notes)
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity

def beat_extraction(midi_file):
    """
    Extract beat position and length from a midi file
    
    Args:
        midi_file: Path to a midi file
        
    Returns:
        A list of (beat position, beat length) values
    """
    midi = Score(midi_file)
    tokens = tokenizer(midi)[0].tokens
    
    beats = []
    position = None
    
    for i in range(len(tokens)):
        # Reset position when encountering a bar marker
        if tokens[i] == 'Bar_None':
            position = None
            continue
            
        # Capture position tokens
        if tokens[i].startswith('Position_'):
            position = int(tokens[i].split('_')[1])
        
        # Capture duration tokens if we already have a position
        if position is not None and tokens[i].startswith('Duration_'):
            duration_token = tokens[i].split('_')[1]
            if duration_token in duration2length:
                beat_length = duration2length[duration_token]
                beats.append((position, beat_length))
    
    return beats

def beat_bigram_probability(midi_files):
    """
    Calculate bigram probabilities p(beat_length | previous_beat_length)
    
    Args:
        midi_files: List of paths to midi files
        
    Returns:
        Two dictionaries:
        - bigramBeatTransitions: maps previous_beat_length to list of beat_lengths
        - bigramBeatTransitionProbabilities: maps previous_beat_length to list of beat_length probabilities
    """
    bigramBeatTransitions = defaultdict(list)
    bigramBeatTransitionProbabilities = defaultdict(list)
    
    # Count bigram occurrences
    bigram_counts = defaultdict(lambda: defaultdict(int))
    
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        
        # Extract just the beat lengths
        beat_lengths = [length for _, length in beats]
        
        # Count transitions between consecutive beat lengths
        for i in range(len(beat_lengths) - 1):
            prev_length = beat_lengths[i]
            next_length = beat_lengths[i + 1]
            bigram_counts[prev_length][next_length] += 1
    
    # Convert counts to probabilities
    for prev_length, next_lengths in bigram_counts.items():
        total = sum(next_lengths.values())
        
        # Create ordered lists of next beat lengths and their probabilities
        for next_length, count in next_lengths.items():
            bigramBeatTransitions[prev_length].append(next_length)
            bigramBeatTransitionProbabilities[prev_length].append(count / total)
    
    return bigramBeatTransitions, bigramBeatTransitionProbabilities

def beat_pos_bigram_probability(midi_files):
    """
    Calculate bigram probabilities p(beat_length | beat_position)
    
    Args:
        midi_files: List of paths to midi files
        
    Returns:
        Two dictionaries:
        - bigramBeatPosTransitions: maps beat_position to list of beat_lengths
        - bigramBeatPosTransitionProbabilities: maps beat_position to list of beat_length probabilities
    """
    bigramBeatPosTransitions = defaultdict(list)
    bigramBeatPosTransitionProbabilities = defaultdict(list)
    
    # Count position-length occurrences
    position_length_counts = defaultdict(lambda: defaultdict(int))
    
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        
        # Count occurrences of beat lengths at specific positions
        for position, length in beats:
            position_length_counts[position][length] += 1
    
    # Convert counts to probabilities
    for position, lengths in position_length_counts.items():
        total = sum(lengths.values())
        
        # Create ordered lists of beat lengths and their probabilities
        for length, count in lengths.items():
            bigramBeatPosTransitions[position].append(length)
            bigramBeatPosTransitionProbabilities[position].append(count / total)
    
    return bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities

def beat_bigram_perplexity(midi_file):
    """
    计算两个模型的困惑度
    
    参数:
        midi_file: MIDI文件路径
    
    返回:
        perplexity_Q7: 基于前一个节拍长度的模型的困惑度
        perplexity_Q8: 基于节拍位置的模型的困惑度
    """
    # 获取两个概率模型
    bigramBeatTransitions, bigramBeatTransitionProbabilities = beat_bigram_probability(midi_files)
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    
    # 计算节拍长度的一元概率分布(用于平滑处理)
    beat_length_counts = defaultdict(int)
    total_beats = 0
    
    for mf in midi_files:
        beats = beat_extraction(mf)
        for _, length in beats:
            beat_length_counts[length] += 1
            total_beats += 1
    
    beat_length_probabilities = {length: count / total_beats for length, count in beat_length_counts.items()}
    
    # 获取节拍信息
    beat_info = beat_extraction(midi_file)
    N = len(beat_info)
    
    if N == 0:
        return float('inf'), float('inf')
    
    # 计算两个模型的对数概率之和
    log_prob_sum_Q7 = 0
    log_prob_sum_Q8 = 0
    
    # 第一个节拍的一元概率(用于Q7)
    if N > 0:
        first_beat_prob = beat_length_probabilities.get(beat_info[0][1], 1e-10)
        log_prob_sum_Q7 += np.log(first_beat_prob)
    
    # 计算有效样本数
    effective_count_Q7 = 0
    effective_count_Q8 = 0
    
    for i in range(N):
        position, length = beat_info[i]
        
        # 计算Q7模型的对数概率(基于前一个节拍长度)
        if i > 0:
            effective_count_Q7 += 1
            prev_length = beat_info[i-1][1]
            if prev_length in bigramBeatTransitions:
                if length in bigramBeatTransitions[prev_length]:
                    idx = bigramBeatTransitions[prev_length].index(length)
                    log_prob_sum_Q7 += np.log(bigramBeatTransitionProbabilities[prev_length][idx])
                else:
                    # 平滑处理
                    smooth_prob = beat_length_probabilities.get(length, 1e-10)
                    log_prob_sum_Q7 += np.log(smooth_prob)
            else:
                # 平滑处理
                smooth_prob = beat_length_probabilities.get(length, 1e-10)
                log_prob_sum_Q7 += np.log(smooth_prob)
        
        # 计算Q8模型的对数概率(基于节拍位置)
        effective_count_Q8 += 1
        if position in bigramBeatPosTransitions:
            if length in bigramBeatPosTransitions[position]:
                idx = bigramBeatPosTransitions[position].index(length)
                log_prob_sum_Q8 += np.log(bigramBeatPosTransitionProbabilities[position][idx])
            else:
                # 平滑处理
                smooth_prob = beat_length_probabilities.get(length, 1e-10)
                log_prob_sum_Q8 += np.log(smooth_prob)
        else:
            # 平滑处理
            smooth_prob = beat_length_probabilities.get(length, 1e-10)
            log_prob_sum_Q8 += np.log(smooth_prob)
    
    # 计算困惑度
    # 确保分母不为0
    if effective_count_Q7 > 0:
        perplexity_Q7 = np.exp(-log_prob_sum_Q7 / (effective_count_Q7 + 1))  # +1 是因为第一个节拍
    else:
        perplexity_Q7 = float('inf')
        
    if effective_count_Q8 > 0:
        perplexity_Q8 = np.exp(-log_prob_sum_Q8 / effective_count_Q8)
    else:
        perplexity_Q8 = float('inf')
    
    return perplexity_Q7, perplexity_Q8
def beat_trigram_probability(midi_files):
    """
    Calculate trigram probabilities p(beat_length | previous_beat_length, beat_position)
    
    Args:
        midi_files: List of paths to midi files
        
    Returns:
        Two dictionaries:
        - trigramBeatTransitions: maps (previous_beat_length, beat_position) to list of beat_lengths
        - trigramBeatTransitionProbabilities: maps (previous_beat_length, beat_position) to list of beat_length probabilities
    """
    trigramBeatTransitions = defaultdict(list)
    trigramBeatTransitionProbabilities = defaultdict(list)
    
    # Count trigram occurrences
    trigram_counts = defaultdict(lambda: defaultdict(int))
    
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        
        # Process sequence to find (previous_beat_length, position, beat_length) patterns
        for i in range(1, len(beats)):
            prev_length = beats[i-1][1]  # Length of previous beat
            position = beats[i][0]       # Position of current beat
            length = beats[i][1]         # Length of current beat
            
            trigram_counts[(prev_length, position)][length] += 1
    
    # Convert counts to probabilities
    for (prev_length, position), lengths in trigram_counts.items():
        total = sum(lengths.values())
        
        # Create ordered lists of beat lengths and their probabilities
        for length, count in lengths.items():
            trigramBeatTransitions[(prev_length, position)].append(length)
            trigramBeatTransitionProbabilities[(prev_length, position)].append(count / total)
    
    return trigramBeatTransitions, trigramBeatTransitionProbabilities

def beat_trigram_perplexity(midi_file):
    """
    Calculate perplexity of the trigram beat model on a midi file
    
    Args:
        midi_file: Path to a midi file
        
    Returns:
        Perplexity value
    """
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    trigramBeatTransitions, trigramBeatTransitionProbabilities = beat_trigram_probability(midi_files)
    
    # Compute beat length unigram probabilities
    beat_length_counts = defaultdict(int)
    total_beats = 0
    
    for mf in midi_files:
        beats = beat_extraction(mf)
        for _, length in beats:
            beat_length_counts[length] += 1
            total_beats += 1
    
    beat_length_probabilities = {length: count / total_beats for length, count in beat_length_counts.items()}
    
    # Extract beats from test file
    beats = beat_extraction(midi_file)
    
    if len(beats) <= 1:
        return float('inf')  # Handle edge case of very short sequences
    
    # Calculate log probabilities sum
    log_prob_sum = 0
    
    # Handle first beat using unigram probability
    first_beat_prob = beat_length_probabilities.get(beats[0][1], 1e-10)
    log_prob_sum += np.log(first_beat_prob)
    
    # Handle remaining beats using trigram probabilities
    for i in range(1, len(beats)):
        prev_length = beats[i-1][1]
        position = beats[i][0]
        curr_length = beats[i][1]
        
        if (prev_length, position) in trigramBeatTransitions:
            if curr_length in trigramBeatTransitions[(prev_length, position)]:
                idx = trigramBeatTransitions[(prev_length, position)].index(curr_length)
                prob = trigramBeatTransitionProbabilities[(prev_length, position)][idx]
            else:
                # Smoothing for unseen transitions
                prob = 1e-10
        else:
            # Fall back to position-based bigram if trigram not found
            if position in bigramBeatPosTransitions:
                if curr_length in bigramBeatPosTransitions[position]:
                    idx = bigramBeatPosTransitions[position].index(curr_length)
                    prob = bigramBeatPosTransitionProbabilities[position][idx]
                else:
                    prob = 1e-10
            else:
                # Fall back to unigram if bigram not found
                prob = beat_length_probabilities.get(curr_length, 1e-10)
        
        log_prob_sum += np.log(prob)
    
    # Calculate perplexity
    avg_log_prob = log_prob_sum / len(beats)
    perplexity = np.exp(-avg_log_prob)
    
    return perplexity

def music_generate(length):
    """
    Generate a piece of music using Markov chains and save as MIDI
    
    Args:
        length: Number of notes to generate
    """
    # Models for note generation
    unigramProbabilities = note_unigram_probability(midi_files)
    _, _ = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)
    
    # Models for beat generation
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    
    # Sample initial notes (first two notes)
    notes = list(unigramProbabilities.keys())
    probabilities = list(unigramProbabilities.values())
    
    # Generate first two notes from unigram distribution
    sampled_notes = [np.random.choice(notes, p=probabilities), 
                     np.random.choice(notes, p=probabilities)]
    
    # Generate remaining notes using trigram model
    for _ in range(length - 2):
        next_prev_note = sampled_notes[-2]
        prev_note = sampled_notes[-1]
        
        if (next_prev_note, prev_note) in trigramTransitions:
            next_notes = trigramTransitions[(next_prev_note, prev_note)]
            probs = trigramTransitionProbabilities[(next_prev_note, prev_note)]
            next_note = np.random.choice(next_notes, p=probs)
        else:
            # Fall back to unigram if no trigram transition found
            next_note = np.random.choice(notes, p=probabilities)
        
        sampled_notes.append(next_note)
    
    # Generate beats based on beat position
    sampled_beats = []
    position = 0
    
    for _ in range(length):
        if position in bigramBeatPosTransitions:
            beat_lengths = bigramBeatPosTransitions[position]
            probs = bigramBeatPosTransitionProbabilities[position]
            beat_length = np.random.choice(beat_lengths, p=probs)
        else:
            # If position not seen in training, use a default value like quarter note (8)
            beat_length = 8
        
        sampled_beats.append((position, beat_length))
        
        # Update position for next note, reset to 0 if we reach end of bar
        position += beat_length
        if position >= 32:  # End of bar (32 positions per bar in 4/4 time)
            position = position % 32
    
    # Create MIDI file
    midi = MIDIFile(1)  # One track
    track = 0
    time = 0
    
    # Set tempo
    tempo = 120  # BPM
    midi.addTempo(track, time, tempo)
    
    # Add notes to the MIDI file
    for i in range(len(sampled_notes)):
        note = sampled_notes[i]
        position, beat_length = sampled_beats[i]
        
        # Convert beat length from MidiTok format (8 = quarter note) to MIDIUtil format (1 = quarter note)
        duration = beat_length / 8
        
        # Add the note to the MIDI file
        midi.addNote(track, 0, note, time, duration, 100)  # Channel 0, velocity 100
        
        # Increment time
        time += duration
    
    # Write the MIDI file
    with open("q10.mid", "wb") as output_file:
        midi.writeFile(output_file)
    
    return sampled_notes, sampled_beats