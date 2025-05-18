dataroot = "."

import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import random
import glob
import os


torch.use_deterministic_algorithms(True) # Try to make things less random, though not required

audio_paths = glob.glob(dataroot + "/nsynth_subset/*.wav")
random.seed(0)
random.shuffle(audio_paths)
print(audio_paths)



if not len(audio_paths):
    print("You probably need to set the dataroot folder correctly")



SAMPLE_RATE = 8000 # Very low sample rate, just so things run quickly
N_MFCC = 13
INSTRUMENT_MAP = {'guitar': 0, 'vocal': 1} # Only two classes (also so that things run quickly)
NUM_CLASSES = len(INSTRUMENT_MAP)

def extract_waveform(path):
    # Your code here
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE)
    return waveform


# In[13]:


def extract_label(path):
    # Your code here
    filename = os.path.basename(path)
    name_part = filename.split('_')[0]
    label = INSTRUMENT_MAP[name_part]
    return label


# In[14]:


waveforms = [extract_waveform(p) for p in audio_paths]
labels = [extract_label(p) for p in audio_paths]


class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(2 * N_MFCC, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# In[16]:


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(nnF.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nnF.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nnF.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 2. Extract mfcc features
# 
# `extract_mfcc()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# 
# **Outputs**
# - `feature`: a PyTorch float tensor that represents a concatenation of 13 mean values and 13 standard deviation values
# 
# **Process**
# - Extract feature using `librosa.feature.mfcc`; remember to set the sample rate and n_mfcc
# - Compute 13 mean and 13 standard deviation values
# - Concatenate them together

# In[17]:


def extract_mfcc(w):
    # Your code here:
    # load using librosa.feature.mfcc
    librosa_mfcc = librosa.feature.mfcc(y=w, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    # extract mean and std
    mfcc_mean = np.mean(librosa_mfcc, axis=1)
    mfcc_std = np.std(librosa_mfcc, axis=1)
    # concatenate
    features = np.concatenate((mfcc_mean, mfcc_std))
    # return as torch.FloatTensor
    return torch.FloatTensor(features)


# ## Note:
# 
# The autograder will test that your MFCC features are correct, and it will *also* use them within an ML pipeline. The test_suite can be used to run the full pipeline after you've implemented these functions. If you've implemented your features correctly this should "just work" and you'll be able to upload the trained; this is mostly here just so that you can see how the full pipeline works (which will be useful when you develop your own pipelines for Assignment 1)

# 3. Extract spectrograms
# 
# `extract_spec()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# 
# **Outputs**
# - `feature`: a PyTorch float tensor that contains a spectrogram
# 
# **Process**
# - apply STFT to the given waveform
# - square the absolute values of the complex numbers from the STFT

# In[18]:



def extract_spec(waveform):
    # Your code here
    # load
    stft = librosa.stft(waveform)

    # take squared absolute values
    spec = np.abs(stft) ** 2
    spec = torch.FloatTensor(spec)
    return spec


# 4. Extract mel-spectrograms
# 
# `extract_mel()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# - `n_mels`: number of mel bands
# - `hop_length`: hop length
# 
# **Outputs**
# - `feature`: A PyTorch Float Tensor that contains a mel-spectrogram
# 
# **Process**
# - generate melspectrograms with `librosa.feature.melspectrogram`; make sure to se the sample rate, n_mels, and hop_length
# - convert them to decibel units with `librosa.power_to_db`
# - normalize values to be in the range 0 to 1

# In[19]:


def extract_mel(w, n_mels=128, hop_length=512):
    """Extract mel-spectrogram features from waveform.
    
    Args:
        w (np.ndarray): Input waveform as numpy array
        n_mels (int): Number of mel bands to generate
        hop_length (int): Hop length for STFT
        
    Returns:
        torch.FloatTensor: Normalized mel-spectrogram features
        
    Raises:
        ValueError: If input is empty or invalid
    """
    if not isinstance(w, np.ndarray) or len(w) == 0:
        raise ValueError("Input waveform must be non-empty numpy array")
        
    # Compute mel spectrogram directly from numpy array
    mel_spec = librosa.feature.melspectrogram(
        y=w,
        sr=SAMPLE_RATE,
        n_mels=n_mels,
        hop_length=hop_length
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Safe normalization
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    return torch.FloatTensor(mel_spec_db)


# 5. Extract constant-Q transform
# 
# `extract_q()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# 
# **Outputs**
# - `feature`: A PyTorch Float Tensor that contains a constant-Q transform
# 
# **Process**
# - generate constant-Q transform with `librosa.cqt`; this one will need a higher sample rate (use 16000) to work

# In[20]:



def extract_q(waveform, sample_rate=16000):
    # Generate constant-Q transform
    cqt = librosa.cqt(
        y=waveform,
        sr=sample_rate

    )

    # Take the absolute value (since CQT returns complex numbers)
    cqt_abs = np.abs(cqt)

    # Normalize to 0-1 range
    #cqt_norm = (cqt_abs - cqt_abs.min()) / (cqt_abs.max() - cqt_abs.min())

    return torch.FloatTensor(cqt_abs)


# 6. Pitch shift
# 
# `pitch_shift()`
# 
# **Inputs**
# - `waveform`: an array containing the waveform
# - `n`: number of semitones to shift by (integer, can be positive or negative)
# 
# **Outputs**
# - `waveform`: a pitch-shifted waveform
# 
# **Process**
# - use `librosa.effects.pitch_shift`

# In[21]:


def pitch_shift(w, n):
    # Your code here
    shifted_waveform = librosa.effects.pitch_shift(
        y=w,
        sr=SAMPLE_RATE,
        n_steps=n
    )
    return shifted_waveform


# In[22]:


# Code below augments the datasets

augmented_waveforms = []
augmented_labels = []

for w,y in zip(waveforms,labels):
    augmented_waveforms.append(w)
    augmented_waveforms.append(pitch_shift(w,1))
    augmented_waveforms.append(pitch_shift(w,-1))
    augmented_labels += [y,y,y]


# 7. Extend the model to work for four classes.
# 
# By making data augmentations, or modifying the model architecture, build a model with test accuracy > 0.93

# In[23]:


INSTRUMENT_MAP_7 = {'guitar_acoustic': 0, 'guitar_electronic': 1, 'vocal_acoustic': 2, 'vocal_synthetic': 3}


# In[24]:


NUM_CLASSES_7 = 4


# In[25]:


def extract_label_7(path):
    # Your code here
    filename = os.path.basename(path)
    name_part = filename.split('_')[0]
    name_part2 = filename.split('_')[1]
    if name_part == 'guitar' and name_part2 == 'acoustic':
        label = INSTRUMENT_MAP_7['guitar_acoustic']
    elif name_part == 'guitar' and name_part2 == 'electronic':
        label = INSTRUMENT_MAP_7['guitar_electronic']
    elif name_part == 'vocal' and name_part2 == 'acoustic':
        label = INSTRUMENT_MAP_7['vocal_acoustic']
    elif name_part == 'vocal' and name_part2 == 'synthetic':
        label = INSTRUMENT_MAP_7['vocal_synthetic']
    else:
        raise ValueError("Unknown instrument type")
    return label


# In[26]:


# Select which feature function to use.
# Can be one of the existing ones (e.g. extract_mfcc), or you can write a new one.
feature_func_7 = extract_mfcc


# In[27]:


labels_7 = [extract_label_7(p) for p in audio_paths]


# In[33]:


NUM_CLASSES =4  
class MLPClassifier_7class(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(2 * N_MFCC, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, NUM_CLASSES)  # 注意这里，输出类别改成4
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# In[34]:


# Select which model to use.
# Can use an existing model (e.g. MLPClassifier) or modify it.
# Note that you'll need to copy and (slightly) modify the existing class to handle 4 labels.
model_7 = MLPClassifier()


# In[ ]:
