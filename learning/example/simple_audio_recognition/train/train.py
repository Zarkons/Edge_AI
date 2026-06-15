import os
import pathlib
from absl import app

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def get_secure_save_dir():
    # 1. Primary choice: Check if executed via 'bazel run'
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    
    if workspace_dir:
        actual_save_dir = os.path.join(workspace_dir, "learning/example/simple_audio_recognition/train/build/data")
    else:
        # 2. Universal Fallback: Calculate path dynamically relative to this file's position
        # (This handles standard python runs, VS Code debugging, or bazel tests)
        current_file_path = os.path.abspath(__file__)
        # Adjust the number of parents based on your folder depth to reach the root
        repo_root = pathlib.Path(current_file_path).parents[2] 
        actual_save_dir = os.path.join(repo_root, "learning/example/simple_audio_recognition/train/build/data")
        
    # Ensure the directory actually exists on your Mac before saving files to it
    os.makedirs(actual_save_dir, exist_ok=True)
    return pathlib.Path(actual_save_dir)

def load_data():
    """Load the dataset and split it into training, validation, and test sets."""
    data_dir = get_secure_save_dir()
    
    # 🟢 FIX: Check if the directory is empty by looking for any files inside it
    # True if the folder is empty or doesn't exist
    is_folder_empty = not any(data_dir.iterdir()) if data_dir.exists() else True
    
    if is_folder_empty:
        print(f"Dataset folder is empty. Downloading straight to: {data_dir}")
        tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            # Point to the actual physical hard drive location instead of '.'
            cache_dir=str(data_dir.parent), # Points to .../train/data
            cache_subdir=data_dir.name             # Unpacks inside /mini_speech_commands
        )
    else:
        print(f"🟢 Success! Existing dataset found with files at: {data_dir.absolute()}")
        
    nested_data_dir = data_dir / 'mini_speech_commands_extracted' / 'mini_speech_commands'
    if nested_data_dir.exists():
        print(f"🟢 Success! Pointing to nested data directory at: {nested_data_dir.absolute()}")
        return nested_data_dir
    return data_dir

def get_batch_from_dataset(dataset):
    for audio, label in dataset.take(1):
        return audio, label

def print_tensor_info(audio, label):
    print("Audio shape:", audio.numpy().shape)
    print("Label shape:", label.numpy().shape)
    print("Label:", label.numpy())

def get_datasets(data_dir):
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=seed,
    output_sequence_length=16000,
    subset='both')

    return train_ds, val_ds

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

def plot_waveform(audio_batch, label_batch, label_names):
    plt.figure(figsize=(16, 10))
    rows = 3
    cols = 3
    n = rows * cols
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        audio_signal = audio_batch[i]
        plt.plot(audio_signal)
        plt.title(label_names[label_batch[i]])
        plt.yticks(np.arange(-1.2, 1.2, 0.2))
        plt.ylim([-1.1, 1.1])

    workspace_dir = get_secure_save_dir()
    
    if workspace_dir:
        # Save it directly into your physical project root directory folder
        save_destination = os.path.join(workspace_dir.parent, "audio_waveforms_grid.png")
    else:
        save_destination = "audio_waveforms_grid.png"

    plt.savefig(save_destination)

def main(_):
    data_dir = load_data()

    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
    print('Commands:', commands)

    train_ds, val_ds = get_datasets(data_dir)

    label_names = np.array(train_ds.class_names)
    print()
    print("label names:", label_names)
    audio_batch, label_batch = get_batch_from_dataset(train_ds)
    print_tensor_info(audio_batch, label_batch)

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
    audio_batch, label_batch = get_batch_from_dataset(train_ds)
    print_tensor_info(audio_batch, label_batch)

    plot_waveform(audio_batch, label_batch, label_names)

    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    

if __name__ == "__main__":
    app.run(main)   