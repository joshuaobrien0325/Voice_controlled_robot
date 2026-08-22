import numpy as np
import os
import keras
import librosa
import tensorflow as tf
import websockets
import asyncio
import matplotlib.pyplot as plt



# Config
file_directory = r"C:\Engineering Final Project"
sampling_rate  = 16000  # sampling rate 16,000 samples per second 16,000HZ
duration       = 1      # duration of the words set to 1 second
confidence_threshold = 0.75  # requires a threshold of 0.75 or above to match a word to the audio
word_to_char_map = { # converting words into corresponding charcters
    "up": "U",
    "down": "D",
    "left": "L",
    "right": "R",
    "stop": "S",
    "go": "G",
    "yes": "Y",
    "no": "N"
}
# Loading in all cnn files
model = keras.models.load_model(
    os.path.join(file_directory, "final_cnn.keras"))
cnn_norm    = np.load(os.path.join(file_directory, "final_cnn_norm.npy"))
class_names = np.load(os.path.join(file_directory, "final_cnn_class_names.npy"),
                      allow_pickle=True)

cnn_mean, cnn_std, max_len = cnn_norm
max_len = int(max_len)

print(f"Model loaded — commands: {list(class_names)}")


# this function extracts the mel spectrogram of the input audio
def extract_mel(audio_array, sampling_rate):
    if audio_array.dtype != np.float64: # forces any integer value into float to avoid math errors
        audio_array = audio_array.astype(np.float64)

    # Pre emphasis which applies a high pass filter to amplify high frequency speech components
    pre_emphasis = 0.95
    emphasized = np.append(audio_array[0],
                           audio_array[1:] - pre_emphasis
                           * audio_array[:-1])

    # Framing
    frame_length = int(round(0.025 * sampling_rate)) # calculates the number of samples in each frame
    frame_stride = int(round(0.010 * sampling_rate)) # calculates the number of samples of each frame stride
    signal_length = len(emphasized)
    num_frames = 1 + int(np.ceil(
        float(np.abs(signal_length - frame_length)) / frame_stride)) # calculates how many frames are needed for the whole signal

    # Padding
    pad_length = num_frames * frame_stride + frame_length # since the frames round up to the nearest integer it is neccessary to pad the extra space with zeros
    padded = np.append(emphasized, np.zeros(pad_length - signal_length))

    #this section creates an array where each row is a frame of audio
    indices = (np.tile(np.arange(0, frame_length), (num_frames, 1)) +
               np.tile(np.arange(0, num_frames * frame_stride,
                                 frame_stride),
                       (frame_length, 1)).T)
    frames = padded[indices.astype(np.int32)]
    frames *= np.hamming(frame_length) # applies a hamming window to account for spectral leakage

    # Fast fourier tranform
    NFFT = 512 # uses a 512 point fft outputting 257 bins
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT)) # gets the magnitude spectrum from the fft
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2) # converts magnitude spectrum to power spectrum

    # This section converts the power spectrum to mel spectogram
    nfilt    = 40
    low_mel  = 0
    high_mel = 2595 * np.log10(1 + (sampling_rate / 2) / 700)
    mel_points = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_points  = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((NFFT + 1) * hz_points / sampling_rate)



    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bins[m - 1])
        f_m       = int(bins[m])
        f_m_plus  = int(bins[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k - bins[m-1]) / (bins[m] - bins[m-1])
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (bins[m+1] - k) / (bins[m+1] - bins[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0,
                            np.finfo(float).eps, filter_banks)



    return 20 * np.log10(filter_banks)



# function to predict the most likely word
def predict(audio_array, sampling_rate):
    mel = extract_mel(audio_array, sampling_rate)

    # Corrects the size to be the same size as CNN was trained on
    if mel.shape[0] < max_len:
        mel = np.pad(mel, ((0, max_len - mel.shape[0]), (0, 0)),
                     mode='constant')
    else:
        mel = mel[:max_len, :]

    mel_input = mel[np.newaxis, ..., np.newaxis].astype(np.float32)

    # Predict
    logits = model.predict(mel_input, verbose=0)[0] # runs the cnn and returns a score for each word
    probs = tf.nn.softmax(logits).numpy() # converts the score from cnn into a distribution that sums to 1.0
    pred_class = np.argmax(probs) # finds the i ndex with highest probability
    confidence = float(probs[pred_class])
    word = class_names[pred_class] # maps the index back to a word

    return word, confidence, probs


# function to ensure audio is exact size as training data
def process_audio(audio_array):

    audio_array = audio_array.astype(np.float32)

    # Any audio which is 20Db quieter than peak is trimmed
    audio_trimmed, _ = librosa.effects.trim(audio_array, top_db=20)

    # is spoken word is above 1 second it is cut
    if len(audio_trimmed) >= sampling_rate:
        return audio_trimmed[:sampling_rate]

    # if spoken word is below 1 sec zeros are padded to make it  1 second
    total_pad  = sampling_rate - len(audio_trimmed)
    pad_before = total_pad // 2
    pad_after  = total_pad - pad_before

    return np.pad(audio_trimmed, (pad_before, pad_after), mode='constant')

# function to connect the application via websocket and stream real time audio
async def handle_audio_stream(websocket): # async is used as this allows the func to pause without blocking other tasks
    print(f"\nClient connected from {websocket.remote_address}")
    audio_buffer = bytearray() #the audio will be held in this buffer

    try:
        async for message in websocket: # each message recieved is either a chunk of bytes for audio or an
            # end stream message if an end stream message is sent the audio will be processed otherwise the audio message is added to the buffer
            if isinstance(message, str):
                if message == "END_OF_STREAM":
                    print(" End of audio stream. Processing buffer now.")
                    break
                continue

            # Append incoming binary chunks to our buffer
            audio_buffer.extend(message)

          #calc for 1 second in bytes
            bytes_per_second = sampling_rate * 2

            if len(audio_buffer) >= bytes_per_second:
                # Extract exactly 1 second of audio data
                chunk = audio_buffer[:bytes_per_second]
                # Keep the remainder for the next iteration
                audio_buffer = audio_buffer[bytes_per_second:]

                # converts audio
                audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0

                # check if there is audio in this 1 second chunk
                peak = np.max(np.abs(audio_float32))
                if peak < 0.01:
                    print("No speech detected in chunk")
                    continue
                # processes audio using the function previously created
                processed = process_audio(audio_float32)
                word, confidence, probs = predict(processed, sampling_rate)

                # Send result back to Flutter if a confidence threshold is met
                if confidence >= confidence_threshold:

                    predicted_char = word_to_char_map.get(word.lower(), word[0].upper())

                    print(f"Sending: '{predicted_char}' (from word: {word}, {confidence:.0%})")
                    await websocket.send(predicted_char)
                else:
                    print(f" Low confidence ({confidence:.0%}) for: {word}")

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f" Error: {e}")


async def main():
    # 0.0.0.0 allows any other device to connect on the same network
    # Port 8765 is being used as this is the port flutter is programmed to send traffic to
    server = await websockets.serve(handle_audio_stream, "0.0.0.0", 8765)
    print(" WebSocket Server Started")
    print(" Listening on ws://0.0.0.0:8765")



    await asyncio.Future()  # Runs indefinetly


if __name__ == "__main__":
    asyncio.run(main())