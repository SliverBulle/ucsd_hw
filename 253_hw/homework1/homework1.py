#!/usr/bin/env python
# coding: utf-8

# # Homework 1: Sine wave generation and binary classification

# ## Part A - Sine Wave Generation

# ### Setup
# To complete this part, install the required Python libraries:

# In[7]:


import numpy as np
from scipy.io import wavfile

import numpy as np
import glob
from mido import MidiFile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[8]:


# (installation process may be different on your system)
# You don't need to use these libraries, so long as you implement the specified functions
# !pip install numpy
# !pip install scipy
# !pip install IPython
# !pip install glob
# !pip install scikit-learn
# !pip install mido


# 1. Write a function that converts a musical note name to its corresponding frequency in Hertz (Hz)
# 
# `note_name_to_frequency()`
# - **Input**: A string `note_name` combining a note (e.g., `'C'`, `'C#'`, `'D'`, `'D#'`, `'E'`, `'F'`, `'F#'`, `'G'`, `'G#'`, `'A'`, `'A#'`, `'B'`) and an octave number (`'0'` to `'10'`)
# - **Output**: A float representing the frequency in Hz
# - **Details**:
#   - Use A4 = 440 Hz as the reference frequency
#   - Frequencies double with each octave increase (e.g., A5 = 880 Hz) and halve with each decrease (e.g., A3 = 220 Hz)
# 
# - **Examples**:
#   - `'A4'` → `440.0`
#   - `'A3'` → `220.0`
#   - `'G#4'` → `415.3047`

# In[9]:


SAMPLE_RATE = 44100

def note_name_to_frequency(note_name):
    notes = {'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4, 
             'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2}
    
    if len(note_name) == 2:
        note, octave = note_name[0], int(note_name[1])
    else:
        note, octave = note_name[:2], int(note_name[2])
    
    semitones_from_a4 = notes[note] + (octave - 4) * 12
    
    frequency = 440.0 * (2 ** (semitones_from_a4 / 12))
    
    return frequency


# 2. Write a function that linearly decreases the amplitude of a given waveform
# 
# `decrease_amplitude()`
# - **Inputs**:
#   - `audio`: A NumPy array representing the audio waveform at a sample rate of 44100 Hz
# - **Output**: A NumPy array representing the audio waveform at a sample rate of 44100 Hz
# - **Details**:
#   - The function must linearly decrease the amplitude of the input audio. The amplitude should start at 1 (full volume) and decrease gradually to 0 (silence) by the end of the sample

# In[10]:


def decrease_amplitude(audio):

    amplitude_factor = np.linspace(1.0, 0.0, len(audio))
    

    modified_audio = audio * amplitude_factor
    
    return modified_audio


# 3. Write a function that adds a delay effect to a given audio where the output is a combination of the original audio and a delayed audio
# 
# `add_delay_effects()`  
# - **Inputs**:  
#   - `audio`: A NumPy array representing the audio waveform, sampled at 44,100 Hz
# - **Output**:  
#   - A NumPy array representing the modified audio waveform, sampled at 44,100 Hz
# - **Details**:
#   - The amplitude of the delayed audio should be 30% of the original audio's amplitude
#   - The amplitude of the original audio should be adjusted to 70% of the original audio's amplitude
#   - The output should combine the original audio (with the adjusted amplitude) with a delayed version of itself
#   - The delayed audio should be offset by 0.5 seconds behind the original audio
# 
# - **Examples**:
#   - The provided files (input.wav and output.wav) provide examples of input and output audio

# In[11]:


# Can use these for visualization if you like, though the autograder won't use ipython
#
# from IPython.display import Audio, display
#
# print("Example Input Audio:")
# display(Audio(filename = "input.wav", rate=44100))
# 
# print("Example Output Audio:")
# display(Audio(filename = "output.wav", rate=44100))


# In[12]:


def add_delay_effects(audio):
    # 计算0.5秒对应的样本数量
    delay_samples = int(0.5 * SAMPLE_RATE)
    
    # 创建延迟音频数组（初始为全0）
    delayed_audio = np.zeros(len(audio) + delay_samples)
    
    # 将原始音频（振幅调整为70%）放入结果数组的开始部分
    delayed_audio[:len(audio)] = audio * 0.7
    
    # 将延迟音频（振幅调整为30%）添加到结果数组中，偏移0.5秒
    delayed_audio[delay_samples:delay_samples + len(audio)] += audio * 0.3
    
    return delayed_audio


# 4. Write a function that concatenates a list of audio arrays sequentially and a function that mixes audio arrays by scaling and summing them, simulating simultaneous playback
# 
# `concatenate_audio()`
# - **Input**:
#   - `list_of_your_audio`: A list of NumPy arrays (e.g., `[audio1, audio2]`), each representing audio at 44100 Hz
# - **Output**: A NumPy array of the concatenated audio
# - **Example**:
#   - If `audio1` is 2 seconds (88200 samples) and `audio2` is 1 second (44100 samples), the output is 3 seconds (132300 samples)
# 
# `mix_audio()`
# - **Inputs**:
#   - `list_of_your_audio`: A list of NumPy arrays (e.g., `[audio1, audio2]`), all with the same length at 44100 Hz.
#   - `amplitudes`: A list of floats (e.g., `[0.2, 0.8]`) matching the length of `list_of_your_audio`
# - **Output**: A NumPy array representing the mixed audio
# - **Example**:
#   - If `audio1` and `audio2` are 2 seconds long, and `amplitudes = [0.2, 0.8]`, the output is `0.2 * audio1 + 0.8 * audio2`

# In[13]:


def concatenate_audio(list_of_your_audio):
    # 顺序连接音频数组列表中的所有音频
    # 使用numpy的concatenate函数将所有音频数组按顺序连接起来
    concatenated_audio = np.concatenate(list_of_your_audio)
    
    return concatenated_audio


# In[14]:


def mix_audio(list_of_your_audio, amplitudes):
    # 确保所有音频数组长度相同
    # 初始化混合后的音频数组为全零数组，长度与输入音频相同
    mixed_audio = np.zeros_like(list_of_your_audio[0])
    
    # 按照给定的振幅比例混合所有音频
    for i, audio in enumerate(list_of_your_audio):
        # 将每个音频乘以对应的振幅系数，然后累加到结果中
        mixed_audio += audio * amplitudes[i]
    
    return mixed_audio


# 5. Modify your solution to Q2 so that your pipeline can generate sawtooth waves by adding harmonics based on the following equation:
# 
#     $\text{sawtooth}(f, t) = \frac{2}{\pi} \sum_{k=1}^{19} \frac{(-1)^{k+1}}{k} \sin(2\pi k f t)$ 
# 
# - **Inputs**:
#   - `frequency`: Fundamental frequency of sawtooth wave
#   - `duration`: A float representing the duration in seconds (e.g., 2.0)
# - **Output**: A NumPy array representing the audio waveform at a sample rate of 44100 Hz

# In[15]:


def create_sawtooth_wave(frequency, duration, sample_rate=44100):
    # 创建时间数组
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 初始化锯齿波数组
    wave = np.zeros_like(t)
    
    # 根据公式添加19个谐波
    # sawtooth(f, t) = (2/π) * Σ((-1)^(k+1)/k * sin(2πkft))，k从1到19
    for k in range(1, 20):  # 从1到19
        # 计算第k个谐波
        harmonic = (2/np.pi) * ((-1)**(k+1)/k) * np.sin(2 * np.pi * k * frequency * t)
        # 将谐波添加到锯齿波中
        wave += harmonic
    
    return wave


# ## Part B - Binary Classification
# Train a binary classification model using `scikit-learn` to distinguish between piano and drum MIDI files.

# #### Unzip MIDI Files
# Extract the provided MIDI datasets:
# 
# ```bash
# unzip piano.zip
# unzip drums.zip
# ```
# 
# - `./piano`: Contains piano MIDI files (e.g., `0000.mid` to `2154.mid`)
# - `./drums`: Contains drum MIDI files (e.g., `0000.mid` to `2154.mid`)
# - Source: [Tegridy MIDI Dataset] (https://github.com/asigalov61/Tegridy-MIDI-Dataset)
# 
# These folders should be extracted into the same directory as your solution file

# In[ ]:





# 6. Write functions to compute simple statistics about the files
# 
# ####  `get_stats()`
# 
# - **Inputs**:
#   - `piano_file_paths`: List of piano MIDI file paths`
#   - `drum_file_paths`: List of drum MIDI file paths`
# - **Output**: A dictionary:
#   - `"piano_midi_num"`: Integer, number of piano files
#   - `"drum_midi_num"`: Integer, number of drum files
#   - `"average_piano_beat_num"`: Float, average number of beats in piano files
#   - `"average_drum_beat_num"`: Float, average number of beats in drum files
# - **Details**:
#   - For each file:
#     - Load with `MidiFile(file_path)`
#     - Get `ticks_per_beat` from `mid.ticks_per_beat`
#     - Compute total ticks as the maximum cumulative `msg.time` (delta time) across tracks
#     - Number of beats = (total ticks / ticks_per_beat)
#   - Compute averages, handling empty lists (return 0 if no files)

# In[16]:


def get_file_lists():
    piano_files = sorted(glob.glob("./piano/*.mid"))
    drum_files = sorted(glob.glob("./drums/*.mid"))
    return piano_files, drum_files

def get_num_beats(file_path):
    # 加载MIDI文件
    mid = MidiFile(file_path)
    
    # 获取每拍的tick数
    ticks_per_beat = mid.ticks_per_beat
    
    # 计算每个轨道的总tick数
    total_ticks_per_track = []
    for track in mid.tracks:
        cumulative_ticks = 0
        for msg in track:
            cumulative_ticks += msg.time
        total_ticks_per_track.append(cumulative_ticks)
    
    # 获取最大的累积tick数
    if total_ticks_per_track:
        max_ticks = max(total_ticks_per_track)
        # 计算拍数 = 总tick数 / 每拍tick数
        num_beats = max_ticks / ticks_per_beat
        return num_beats
    else:
        return 0  # 如果没有轨道或轨道为空，返回0

def get_stats(piano_path_list, drum_path_list):
    piano_beat_nums = []
    drum_beat_nums = []
    
    # 计算每个钢琴文件的拍数
    for file_path in piano_path_list:
        piano_beat_nums.append(get_num_beats(file_path))
    
    # 计算每个鼓文件的拍数
    for file_path in drum_path_list:
        drum_beat_nums.append(get_num_beats(file_path))
    
    # 计算平均拍数，处理空列表情况
    avg_piano_beats = 0 if not piano_beat_nums else np.mean(piano_beat_nums)
    avg_drum_beats = 0 if not drum_beat_nums else np.mean(drum_beat_nums)
    
    # 返回统计结果字典
    return {
        "piano_midi_num": len(piano_path_list),
        "drum_midi_num": len(drum_path_list),
        "average_piano_beat_num": avg_piano_beats,
        "average_drum_beat_num": avg_drum_beats
    }


# 7. Implement a few simple feature functions, to compute the lowest and highest MIDI note numbers in a file, and the set of unique notes in a file
# 
# `get_lowest_pitch()` and `get_highest_pitch()`
# functions to find the lowest and highest MIDI note numbers in a file
# 
# - **Input**: `file_path`, a string (e.g., `"./piano/0000.mid"`)
# - **Output**: An integer (0–127) or `None` if no notes exist
# - **Details**:
#   - Use `MidiFile(file_path)` and scan all tracks
#   - Check `msg.type == 'note_on'` and `msg.velocity > 0` for active notes
#   - Return the minimum (`get_lowest_pitch`) or maximum (`get_highest_pitch`) `msg.note`
# 
# `get_unique_pitch_num()`
# a function to count unique MIDI note numbers in a file
# 
# - **Input**: `file_path`, a string
# - **Output**: An integer, the number of unique pitches
# - **Details**:
#   - Collect `msg.note` from all `'note_on'` events with `msg.velocity > 0` into a set
#   - Return the set’s length
# - **Example**: For notes `["C4", "C4", "G4", "G4", "A4", "A4", "G4"]`, output is 3 (unique: C4, G4, A4)

# In[17]:


def get_lowest_pitch(file_path):
    # 加载MIDI文件
    mid = MidiFile(file_path)
    
    # 初始化最低音符为None
    lowest_pitch = None
    
    # 遍历所有轨道和消息
    for track in mid.tracks:
        for msg in track:
            # 检查是否为note_on事件且音符有效（velocity > 0）
            if msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                # 如果是第一个音符或比当前最低音符更低
                if lowest_pitch is None or msg.note < lowest_pitch:
                    lowest_pitch = msg.note
    
    return lowest_pitch  # 如果没有找到音符，返回None

def get_highest_pitch(file_path):
    # 加载MIDI文件
    mid = MidiFile(file_path)
    
    # 初始化最高音符为None
    highest_pitch = None
    
    # 遍历所有轨道和消息
    for track in mid.tracks:
        for msg in track:
            # 检查是否为note_on事件且音符有效（velocity > 0）
            if msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                # 如果是第一个音符或比当前最高音符更高
                if highest_pitch is None or msg.note > highest_pitch:
                    highest_pitch = msg.note
    
    return highest_pitch  # 如果没有找到音符，返回None

def get_unique_pitch_num(file_path):
    # 加载MIDI文件
    mid = MidiFile(file_path)
    
    # 创建一个集合来存储唯一的音符
    unique_pitches = set()
    
    # 遍历所有轨道和消息
    for track in mid.tracks:
        for msg in track:
            # 检查是否为note_on事件且音符有效（velocity > 0）
            if msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                # 将音符添加到集合中
                unique_pitches.add(msg.note)
    
    # 返回唯一音符的数量
    return len(unique_pitches)


# 8. Implement an additional feature extraction function to compute the average MIDI note number in a file
# 
# `get_average_pitch_value()`
# a function to return the average MIDI note number from a file
# 
# - **Input**: `file_path`, a string
# - **Output**: A float, the average value of MIDI notes in the file
# - **Details**:
#   - Collect `msg.note` from all `'note_on'` events with `msg.velocity > 0` into a set
# - **Example**: For notes `[51, 52, 53]`, output is `52`

# In[18]:


def get_average_pitch_value(file_path):
    # 加载MIDI文件
    mid = MidiFile(file_path)
    
    # 创建一个列表来存储所有音符
    all_pitches = []
    
    # 遍历所有轨道和消息
    for track in mid.tracks:
        for msg in track:
            # 检查是否为note_on事件且音符有效（velocity > 0）
            if msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                # 将音符添加到列表中
                all_pitches.append(msg.note)
    
    # 计算平均值
    # 如果没有找到音符，返回None
    if all_pitches:
        return sum(all_pitches) / len(all_pitches)
    else:
        return None


# 9. Construct your dataset and split it into train and test sets using `scikit-learn` (most of this code is provided). Train your model to classify whether a given file is intended for piano or drums.
# 
# `featureQ9()`
# 
# Returns a feature vector concatenating the four features described above
# 
# - **Input**: `file_path`, a string.
# - **Output**: A vector of four features

# In[19]:


def featureQ9(file_path):
    # Already implemented: this one is a freebie if you got everything above correct!
    return [get_lowest_pitch(file_path),
            get_highest_pitch(file_path),
            get_unique_pitch_num(file_path),
            get_average_pitch_value(file_path)]


# 10. Creatively incorporate additional features into your classifier to make your classification more accurate.  Include comments describing your solution.

# In[20]:


def featureQ10(file_path):
    # 获取基本特征
    base_features = featureQ9(file_path)
    
    # 加载MIDI文件
    mid = MidiFile(file_path)
    
    try:
        # 1. 计算音符范围（最高音符与最低音符的差值）
        pitch_range = float(base_features[1] - base_features[0])
        
        # 2. 收集所有音符和力度值
        notes = []
        velocities = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                    notes.append(msg.note)
                    velocities.append(msg.velocity)
        
        # 3. 计算额外特征
        # 平均力度
        avg_velocity = float(np.mean(velocities)) if velocities else 0.0
        # 音符变化（相邻音符的平均间隔）
        note_changes = [abs(notes[i] - notes[i-1]) for i in range(1, len(notes))]
        avg_note_change = float(np.mean(note_changes)) if note_changes else 0.0
        
        # 4. 返回所有特征
        return  [
            base_features[0],
            base_features[1],
            base_features[2],
            base_features[3],
            pitch_range,         # 音符范围
            avg_velocity,        # 平均力度
            avg_note_change     # 平均音符变化
        ]
    
    except Exception as e:
        # 如果出现任何错误，返回全零特征向量
        return [0.0] * 3 # 基础特征4个 + 新增特征3个

# In[ ]:




