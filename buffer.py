import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import threading
import whisper

from collections import deque


# Constants for audio
FORMAT = pyaudio.paInt16  # 16-bit resolution
CHANNELS = 1  # Mono audio
RATE = 44100  # 44.1 kHz sample rate
CHUNK = 1024  # Number of samples per frame
BUFFER_DURATION = 10  # Ring buffer duration in seconds
WHISPER_RATE = 16000  # Whisper's sample rate

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

        self.model = whisper.load_model("base")

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

        fig, axs = plt.subplots(2, 1, figsize=(14, 8))
        ax1, ax2 = axs

        last_msg = None

        try:
            while self.active:
                with self.buffer_lock:
                    if not self.ring_buffer:
                        continue
                    raw_audio_data = np.concatenate(self.ring_buffer)

                try:
                    # Resample audio to Whisper's expected rate (16 kHz)
                    audio_data = np.interp(
                        np.linspace(
                            0,
                            len(raw_audio_data),
                            num=int(len(raw_audio_data) * WHISPER_RATE / RATE),
                        ),
                        np.arange(len(raw_audio_data)),
                        raw_audio_data,
                    ).astype("float32")

                    # Create time axis for waveform
                    time_data = np.linspace(
                        0, len(audio_data) / WHISPER_RATE, num=len(audio_data)
                    )

                    audio = whisper.pad_or_trim(audio_data)
                    mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
                    mel_numpy = mel.cpu().numpy()
                    has_spectrogram = True
                except Exception as e:
                    if str(e) != last_msg:
                        print(e)
                    last_msg = str(e)
                    has_spectrogram = False

                # Plot waveform
                ax1.clear()
                ax1.plot(time_data, audio_data)
                ax1.set_title("Audio Data")
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Amplitude")
                ax1.set_xlim([0, BUFFER_DURATION])

                # Plot spectrogram
                ax2.clear()
                if has_spectrogram:
                    ax2.imshow(
                        mel_numpy,
                        aspect="auto",
                        origin="lower",
                        interpolation="nearest",
                    )
                ax2.set_title("Spectrogram")
                ax2.set_xlabel("Time (s)")
                ax2.set_xlim([0, 100 * BUFFER_DURATION])

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
        self.visualize_audio()
        self.audio_thread.join()


manager = AudioManager()
manager.start()
