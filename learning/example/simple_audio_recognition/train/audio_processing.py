
import tensorflow as tf
import numpy as np

def short_time_fourier_transform(waveform):
    """
    Computes the Short-Time Fourier Transform (STFT) of a 1D audio waveform.
    Input: waveform - A 1D tensor of shape [samples] representing the audio signal
    Output: A 2D tensor representing the magnitude of the STFT (spectrogram)
    """
    stft = tf.signal.stft(
        waveform,
        frame_length=255,
        frame_step=128
    )
    return tf.abs(stft)

def get_spectrogram_3D(waveform):
    """
    Converts a 1D float32 audio waveform into a 3D spectrogram matrix.
    Input shape: [samples]
    Output shape: [time, frequency, channels] (3D tensor)
    """
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = short_time_fourier_transform(waveform)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_spectrogram_4D(waveform):
    """
    Converts a 1D/2D float32 audio waveform into a 4D spectrogram matrix.
    Input shape: [1, 16000] or [16000]
    Output shape: [1, 124, 129, 1] (Fully matching Keras Conv2D expectations)
    """
    # 1. Ensure the waveform is floating-point data
    waveform = tf.cast(waveform, tf.float32)

    # 2. If it doesn't have a batch dimension, flatten it and squeeze to 1D
    # tf.signal.stft natively expects a 1D vector or [Batch, Samples]
    waveform = tf.squeeze(waveform)

    spectrogram = short_time_fourier_transform(waveform)

    # 5. Add the mandatory Batch dimension and Channel dimension to make it 4D
    # [124, 129] -> [1, 124, 129, 1]
    spectrogram = spectrogram[tf.newaxis, ...]   # Add Batch dimension at front
    spectrogram = spectrogram[..., tf.newaxis]  # Add Channel dimension at back

    return spectrogram

def prepare_waveform_input(wav_path: str) -> np.ndarray:
    """Loads a .wav file and preprocesses it into a 1D numpy array of shape [1, 16000]."""
    # Read the binary file directly from disk using native TF
    audio_binary = tf.io.read_file(str(wav_path))

    # Decode wav directly (this handles float32 normalization between -1.0 and 1.0 automatically)
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=16000)
    audio = tf.squeeze(audio, axis=-1) # Drop channel dimension -> shape

    # Explicit padding guarantee (in case the file was shorter than 16000)
    audio = audio[:16000]
    if tf.shape(audio)[0] < 16000:
        zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], axis=0)

    # Convert back to numpy array and add batch dimension -> shape [1, 16000]
    return audio.numpy()[np.newaxis, :]