import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import mlx_whisper

from collections import deque


# Constants for audio
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Mono audio
SAMPLE_RATE = 44100  # 44.1 kHz sample rate
CHUNK = 1024  # Number of samples per frame
BUFFER_DURATION = 30  # Ring buffer duration in seconds
WHISPER_RATE = 16000  # Whisper's sample rate

# Calculate the buffer size in chunks
BUFFER_SIZE = int(SAMPLE_RATE / CHUNK * BUFFER_DURATION)

print(BUFFER_DURATION, BUFFER_SIZE)


class AudioManager:
    def __init__(self):
        self.active = True

        # Initialize the ring buffer
        self.ring_buffer = deque(maxlen=BUFFER_SIZE)
        self.ring_buffer_lock = threading.Lock()

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Open a stream
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        self.audio_buffers = []
        self.audio_buffers_lock = threading.Lock()

    def read_audio(self):
        try:
            while self.active:
                # Read audio data from the microphone
                data = self.stream.read(CHUNK, exception_on_overflow=False)

                # Convert data to numpy array for processing if needed
                audio_frame = np.frombuffer(data, dtype=np.int16)

                # Add the frame to the ring buffer
                with self.ring_buffer_lock:
                    self.ring_buffer.append(audio_frame)

        except KeyboardInterrupt:
            self.active = False

        finally:
            # Cleanup resources
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

    def decode_audio(self):
        while self.active:
            with self.ring_buffer_lock:
                if not self.ring_buffer:
                    continue
                raw_audio_data = np.concatenate(self.ring_buffer)
                if len(raw_audio_data) < SAMPLE_RATE * 2:
                    continue
                self.ring_buffer.clear()

            try:
                print("got data!", len(raw_audio_data))
                audio_data = np.interp(
                    np.linspace(
                        0,
                        len(raw_audio_data),
                        num=int(len(raw_audio_data) * WHISPER_RATE / SAMPLE_RATE),
                    ),
                    np.arange(len(raw_audio_data)),
                    raw_audio_data,
                ).astype("float32")

                result = mlx_whisper.transcribe(
                    audio_data,
                    path_or_hf_repo="mlx-community/whisper-large-mlx-4bit",
                )

                print(f"Text:{result["text"]}")
                time.sleep(1)
            except Exception as e:
                print(e)
                time.sleep(1)

    def visualize_audio(self):
        plt.ion()

        fig, axs = plt.subplots(2, 1, figsize=(14, 8))
        ax1, ax2 = axs

        try:
            while self.active:
                with self.ring_buffer_lock:
                    if not self.ring_buffer:
                        continue
                    audio_data = np.concatenate(self.ring_buffer)

                # Create time axis for waveform
                time_data = np.linspace(
                    0, len(audio_data) / SAMPLE_RATE, num=len(audio_data)
                )

                # Plot waveform
                ax1.clear()
                ax1.plot(time_data, audio_data)
                ax1.set_title("Audio Data")
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Amplitude")
                ax1.set_xlim([0, BUFFER_DURATION])

                # Plot spectrogram
                ax2.clear()

                plt.tight_layout()
                plt.pause(0.05)

                if not plt.fignum_exists(fig.number):
                    self.active = False
                    break

        except KeyboardInterrupt:
            self.active = False

    def start(self):
        self.audio_thread = threading.Thread(target=self.read_audio)
        self.audio_thread.start()
        self.decode_thread = threading.Thread(target=self.decode_audio)
        self.decode_thread.start()
        self.visualize_audio()
        self.audio_thread.join()
        self.decode_thread.join()


manager = AudioManager()
manager.start()
