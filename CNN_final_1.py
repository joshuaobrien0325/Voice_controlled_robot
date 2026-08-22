import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from keras import layers
from keras import models

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Opens the directory which holds all the audio files from googles speech dataset v2
data_dir = pathlib.Path(r"C:\Engineering Final Project\Raw_audio_files")

# Define the 10 target commands to train on
target_commands = ['stop', 'left', 'right', 'forward', 'backward',
                   'one', 'two', 'three', 'four', 'five']

# Load the full dataset from all folders
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both')

# Get the full list of class names assigned by the dataset loader
all_label_names = np.array(train_ds.class_names)
print("All label names:", all_label_names)

# Build a set of the original label indices that correspond to target commands
target_indices = set()
for i, name in enumerate(all_label_names):
    if name in target_commands:
        target_indices.add(i)
print(f"Keeping {len(target_indices)} classes: "
      f"{[all_label_names[i] for i in sorted(target_indices)]}")

# Create a lookup table that remaps old label indices to new sequential ones
# and marks non-target classes with -1 so they can be filtered out
old_to_new = np.full(len(all_label_names), -1, dtype=np.int32)
new_label_names = []
for new_idx, old_idx in enumerate(sorted(target_indices)):
    old_to_new[old_idx] = new_idx
    new_label_names.append(all_label_names[old_idx])
label_names = np.array(new_label_names)
remap_table = tf.constant(old_to_new, dtype=tf.int32)

print("Filtered label names:", label_names)


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def filter_and_remap(audio, label):
    """Keep only target commands and remap labels to 0..N-1."""
    new_label = tf.gather(remap_table, label)
    return audio, new_label


train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# Filter to only target commands then remap labels
train_ds = (train_ds
            .unbatch()
            .map(filter_and_remap, tf.data.AUTOTUNE)
            .filter(lambda audio, label: label >= 0)
            .batch(64))
val_ds = (val_ds
          .unbatch()
          .map(filter_and_remap, tf.data.AUTOTUNE)
          .filter(lambda audio, label: label >= 0)
          .batch(64))

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

for example_audio, example_labels in train_ds.take(1):
    print(example_audio.shape)
    print(example_labels.shape)


def retrieve_mel_spectrogram(audio_array, sampling_rate=16000):
    if hasattr(audio_array, 'numpy'):
        audio_array = audio_array.numpy()

    audio_array = audio_array.astype(np.float64)

    # Pre-emphasis applies a difference filter to boost high frequencies
    pre_emphasis_coefficient = 0.95
    Emphasized_audio_array = np.append(
        audio_array[0],
        audio_array[1:] - pre_emphasis_coefficient * audio_array[:-1])

    # Framing: split audio signal into frames of 25ms
    Frame_length = 0.025
    Frame_stride = 0.010

    Frame_length = int(round(Frame_length * sampling_rate))
    Frame_stride = int(round(Frame_stride * sampling_rate))
    signal_length = len(Emphasized_audio_array)

    num_frames = 1 + int(np.ceil(
        float(np.abs(signal_length - Frame_length)) / Frame_stride))

    # Padding to ensure each frame is the same size
    pad_signal_length = num_frames * Frame_stride + Frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(Emphasized_audio_array, z)

    # Create 2D matrix where each row is a frame of audio data
    indices = (np.tile(np.arange(0, Frame_length), (num_frames, 1)) +
               np.tile(np.arange(0, num_frames * Frame_stride, Frame_stride),
                       (Frame_length, 1)).T)
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Apply hamming window to avoid spectral leakage
    frames *= np.hamming(Frame_length)

    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sampling_rate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin = np.floor((NFFT + 1) * hz_points / sampling_rate)

    # Create the 40 overlapping triangular filters of the mel filter bank
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(
        filter_banks == 0, np.finfo(float).eps, filter_banks)
    mel_spectrogram = 20 * np.log10(filter_banks)

    return mel_spectrogram


def spec_ds(ds):
    def map_fn(audio, label):
        spectrogram = tf.py_function(
            func=retrieve_mel_spectrogram,
            inp=[audio],
            Tout=tf.float32)
        spectrogram.set_shape([99, 40])
        spectrogram = tf.expand_dims(spectrogram, axis=-1)
        return spectrogram, label
    return ds.unbatch().map(
        map_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(64)


train_spectrogram_ds = spec_ds(train_ds)
val_spectrogram_ds = spec_ds(val_ds)
test_spectrogram_ds = spec_ds(test_ds)

for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
    break

train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(1000).prefetch(
    tf.data.AUTOTUNE)
val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

input_shape = example_spectrograms.shape[1:]
print('Input shape:', input_shape)
num_labels = len(label_names)
print(f"Number of command classes: {num_labels}")

# normalization layer
norm_layer = layers.Normalization()
norm_layer.adapt(
    data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Resizing(32, 32),
    norm_layer,
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

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(
        r"C:\Users\joshl\OneDrive\Documents\Engineering Final Project",
        "tensorboard_logs"),
    histogram_freq=1)

EPOCHS = 10
print("\nTraining has begun.")
history = model.fit(
    train_spectrogram_ds,
    validation_data=val_spectrogram_ds,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        tensorboard_callback
    ],
)

# Evaluation on test set
print("\nEvaluating on test set")
test_loss, test_acc = model.evaluate(test_spectrogram_ds, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss:     {test_loss:.4f}")

# Classification report
from sklearn.metrics import classification_report

y_pred = model.predict(test_spectrogram_ds)
y_pred_classes = tf.argmax(y_pred, axis=1).numpy()
y_true = tf.concat(
    list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0).numpy()

print("\nClassification Report:")
print(classification_report(
    y_true, y_pred_classes, target_names=list(label_names)))

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('CNN — Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('CNN — Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(
        r"C:\Users\joshl\OneDrive\Documents\Engineering Final Project",
        "final_cnn_training.png"),
    dpi=150)
plt.show()

# Confusion matrix
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=label_names,
            yticklabels=label_names,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.title('Confusion Matrix')
plt.show()

#  Single sample prediction bar chart
x = data_dir / 'stop' / os.listdir(data_dir / 'stop')[0]
x = tf.io.read_file(str(x))
x, sample_rate = tf.audio.decode_wav(x, desired_channels=1,
                                     desired_samples=16000)
x = tf.squeeze(x, axis=-1)
x = retrieve_mel_spectrogram(x)
x = np.expand_dims(x, axis=0).astype(np.float32)

prediction = model(x)
plt.bar(label_names, tf.nn.softmax(prediction[0]))
plt.title('Stop')
plt.ylabel('Probability')
plt.show()
# saves the model as well as norm and class names
save_dir = r"C:\Engineering Final Project"

np.save(os.path.join(save_dir, "final_cnn_norm.npy"),
        np.array([float(norm_layer.mean.numpy().mean()),
                  float(norm_layer.variance.numpy().mean() ** 0.5),
                  99]))
np.save(os.path.join(save_dir, "final_cnn_class_names.npy"), label_names)
print("Norm and class names saved")

model.save(os.path.join(save_dir, "final_cnn.keras"))
print("CNN model saved.")
model.summary()