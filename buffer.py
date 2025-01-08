import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import whisper

from collections import deque


# Constants for audio
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Mono audio
RATE = 44100  # 44.1 kHz sample rate
CHUNK = 1024  # Number of samples per frame
BUFFER_DURATION = 10  # Ring buffer duration in seconds

# Calculate the buffer size in chunks
BUFFER_SIZE = int(RATE / CHUNK * BUFFER_DURATION)


class AudioManager:
    def __init__(self):
        self.active = True

        # Initialize the ring buffer
        self.ring_buffer = deque(maxlen=BUFFER_SIZE)
        self.buffer_lock = threading.Lock()

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Open a stream
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

    def read_audio(self):
        try:
            while self.active:
                # Read audio data from the microphone
                data = self.stream.read(CHUNK, exception_on_overflow=False)

                # Convert data to numpy array for processing if needed
                audio_frame = np.frombuffer(data, dtype=np.int16)

                # Add the frame to the ring buffer
                with self.buffer_lock:
                    self.ring_buffer.append(audio_frame)

        except KeyboardInterrupt:
            self.active = False

        finally:
            # Cleanup resources
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

    def visualize_audio(self):
        plt.ion()
        fig, ax = plt.subplots()
        try:
            while self.active:
                with self.buffer_lock:
                    if not self.ring_buffer:
                        continue
                    audio_data = np.concatenate(self.ring_buffer)
                time_data = np.linspace(0, len(audio_data) / RATE, num=len(audio_data))

                ax.clear()
                ax.set_title("Audio Data")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_xlim([0, BUFFER_DURATION])
                ax.plot(time_data, audio_data)
                plt.pause(0.1)

                if not plt.fignum_exists(fig.number):
                    self.active = False
                    break

        except KeyboardInterrupt:
            self.active = False

    def start(self):
        self.audio_thread = threading.Thread(target=self.read_audio)
        self.audio_thread.start()
        self.visualize_audio()
        self.audio_thread.join()


manager = AudioManager()
manager.start()
