import numpy as np
import lameenc

RATE = 44100

def save_beep(filename, duration, freq):
    t = np.arange(int(RATE * duration)) / RATE
    wave = 0.3 * np.sin(2 * np.pi * freq * t)

    # Smooth start/end to avoid clicking
    fade = min(200, len(wave) // 2)
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)

    # Convert to 16-bit PCM
    pcm = (wave * 32767).astype(np.int16)

    # Encode MP3
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(192)
    encoder.set_in_sample_rate(RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)

    mp3_data = encoder.encode(pcm.tobytes())
    mp3_data += encoder.flush()

    with open(filename, "wb") as f:
        f.write(mp3_data)

    print(f"Saved {filename}")


# Short beep for 3 and 2
save_beep("sounds/small_beep.mp3", 0.1, 800)

# Long beep for 1
save_beep("sounds/long_beep.mp3", 0.5, 1200)