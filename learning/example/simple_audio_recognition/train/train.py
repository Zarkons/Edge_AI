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

from learning.example.simple_audio_recognition.train import audio_processing
from learning.example.simple_audio_recognition.train import sample_processing

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def load_data():
    """Load the dataset and split it into training, validation, and test sets."""
    data_dir = sample_processing.get_build_dir() / "data"
    
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

    workspace_dir = sample_processing.get_build_dir()
    
    if workspace_dir:
        # Save it directly into your physical project root directory folder
        save_destination = os.path.join(workspace_dir, "audio_waveforms_grid.png")
    else:
        save_destination = "audio_waveforms_grid.png"

    plt.savefig(save_destination)

def plot_spectrogram(spectrogram_batch, label_batch, label_names):
    plt.figure(figsize=(16, 10))
    rows = 3
    cols = 3
    n = rows * cols
    
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        
        # Extract individual spectrogram
        spectrogram = spectrogram_batch[i]
        
        # Handle channel dimension if present (e.g., shape becomes 2D)
        if len(spectrogram.shape) > 2:
            assert len(spectrogram.shape) == 3
            spectrogram = np.squeeze(spectrogram, axis=-1)
            
        # Convert frequencies to log scale and transpose for correct time x-axis
        log_spec = np.log(spectrogram.T + np.finfo(float).eps)
        
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        
        # Plot the spectrogram heat map
        plt.pcolormesh(X, Y, log_spec)
        plt.title(label_names[label_batch[i]])

    # Coordinate workspace saving logic
    workspace_dir = sample_processing.get_build_dir()
    
    if workspace_dir:
        # Save it directly into your physical project root directory folder
        save_destination = os.path.join(workspace_dir, "audio_spectrograms_grid.png")
    else:
        save_destination = "audio_spectrograms_grid.png"

    plt.savefig(save_destination)
    plt.close() # Recommended to free memory when running scripts in loops

def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio,label: (audio_processing.get_spectrogram_3D(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)

def get_model(input_shape, num_labels, dataset):

    normalization_layer = layers.Normalization()
    normalization_layer.adapt(data=dataset.map(map_func=lambda spec, label: spec))
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        normalization_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
        ])

    model.summary()

    return model

def save_model(model):
    model_save_path = sample_processing.get_build_dir()

    if model_save_path:
        # Save it directly into your physical project root directory folder
        save_destination = os.path.join(model_save_path, "audio_classification_model.keras")
    else:
        save_destination = "audio_classification_model.keras"
    
    model.save(save_destination)

def plot_history(history):
    metrics = history.history
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')

    plt.subplot(1,2,2)
    plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')

def export_trained_keras_model(model):
    """Export only the trained Keras classifier graph.

    Keep waveform-to-spectrogram preprocessing outside the exported model so
    the classifier graph stays quantization-friendly for TFLite conversion.
    """
    export_save_path = sample_processing.get_build_dir()

    if export_save_path:
        save_destination = os.path.join(export_save_path, "exported_trained_keras_model")
    else:
        save_destination = "exported_trained_keras_model"

    tf.saved_model.save(model, str(save_destination))
    return str(save_destination)

def run_inference(model, waveform):
    """Run inference on a single waveform input."""
    spectrogram = audio_processing.get_spectrogram_4D(waveform)
    
    logits = model(spectrogram, training=False)
    
    probabilities = tf.nn.softmax(logits)
    
    return probabilities

def main(_):
    data_dir = load_data()

    train_ds, val_ds = get_datasets(data_dir)

    label_names = np.array(train_ds.class_names)
    sample_processing.save_label_names(label_names.tolist())

    commands = label_names
    print('Commands:', commands)

    print()
    print("label names:", label_names)
    audio_batch, label_batch = get_batch_from_dataset(train_ds)
    print_tensor_info(audio_batch, label_batch)

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
    audio_batch, label_batch = get_batch_from_dataset(train_ds)
    print_tensor_info(audio_batch, label_batch)

    # plot_waveform(audio_batch, label_batch, label_names)

    spectrogram = audio_processing.get_spectrogram_3D(audio_batch)

    for i in range(3):
        label = label_names[label_batch[i]]
        print("Label:", label)
        print("Spectrogram shape:", spectrogram[i].shape)
        print("Waveform shape:", audio_batch[i].shape)

    # plot_spectrogram(spectrogram, label_batch, label_names)

    """ Note: the shard function is used to split the validation dataset into two parts: one for testing and one for validation. However, whenever test_ds or val_ds is loaded, the whole dataset is loaded into memory."""
    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    train_spectrogram_ds = make_spec_ds(train_ds)
    val_spectrogram_ds = make_spec_ds(val_ds)
    test_spectrogram_ds = make_spec_ds(test_ds)

    train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

    input_shape = spectrogram.shape[1:]
    num_labels = len(label_names)

    model = get_model(input_shape, num_labels, train_spectrogram_ds)
    #save_model(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        )

    EPOCHS = 10
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )
    
    plot_history(history)
    model.evaluate(test_spectrogram_ds, return_dict=True)
    # y_pred = model.predict(test_spectrogram_ds)
    # y_pred = tf.argmax(y_pred, axis=1)
    # y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)

    # confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(confusion_mtx, xticklabels=label_names, yticklabels=label_names, annot=True, fmt='g')
    # plt.xlabel('Prediction')
    # plt.ylabel('Label')
    # plt.show()

    x = data_dir/'no/0ab3b47d_nohash_0.wav'
    # x = tf.io.read_file(str(x))
    # x, sample_rate = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
    # x = tf.squeeze(x, axis=-1)
    # waveform = x
    # x = audio_processing.get_spectrogram_3D(x)
    # x = x[tf.newaxis,...]

    # prediction = model(x)
    # x_labels = ['no', 'yes', 'down', 'go', 'left', 'up', 'right', 'stop']
    # plt.bar(x_labels, tf.nn.softmax(prediction[0]))
    # plt.title('No')
    # plt.show()

    exported_model_path = export_trained_keras_model(model)


if __name__ == "__main__":
    app.run(main)   