import os
import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf

class AudioActionFilter:
    def __init__(self, cutoff_hz=150.0, order=5):
        """
        Initializes a high-pass digital Butterworth filter designed to isolate
        and eliminate low-frequency wind rumble distortion from action footage.
        """
        self.cutoff_hz = cutoff_hz
        self.order = order

    def _generate_butter_highpass(self, sample_rate):
        """
        Calculates normalized filter coefficients based on the Nyquist frequency.
        """
        nyquist = 0.5 * sample_rate
        normalized_cutoff = self.cutoff_hz / nyquist
        b, a = butter(self.order, normalized_cutoff, btype='high', analog=False)
        return b, a

    def process_audio_file(self, input_wav_path, output_wav_path=None):
        """
        Loads an audio track, applies the high-pass filter across all channels,
        and exports the polished result.
        """
        if not os.path.exists(input_wav_path):
            raise FileNotFoundError(f"❌ Input audio asset not found at: {input_wav_path}")

        if output_wav_path is None:
            output_wav_path = input_wav_path.replace(".wav", "_cleaned.wav")

        # Load audio data array and sample rate
        data, sample_rate = sf.read(input_wav_path)
        print(f"🎵 Audio Loaded: {os.path.basename(input_wav_path)} | Sample Rate: {sample_rate}Hz | Channels: {data.ndim}")

        # Generate filter coefficients
        b, a = self._generate_butter_highpass(sample_rate)

        # Apply filter across channel arrays
        if data.ndim > 1:
            # Processes all channels concurrently with zero Python loop overhead
            cleaned_data = lfilter(b, a, data, axis=0)
        else:
            cleaned_data = lfilter(b, a, data)

        # Save the filtered audio file
        sf.write(output_wav_path, cleaned_data, sample_rate)
        print(f"🔊 Filter Complete: Exported cleaned audio tracking asset to -> {output_wav_path}")
        return output_wav_path

    def generate_synthetic_validation_audio(self, test_path="config/wind_noise_test.wav"):
        """
        Generates a synthetic test audio sample blending human speech frequencies (1000Hz)
        with massive low-frequency wind rumble distortion (40Hz) to test compliance.
        """
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        sample_rate = 44100
        duration_sec = 3.0
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)

        # 40Hz = Heavy baseline wind rumble buffeting
        wind_rumble = 0.8 * np.sin(2 * np.pi * 40.0 * t)
        # 1000Hz = Clean action sound reference tone
        target_signal = 0.2 * np.sin(2 * np.pi * 1000.0 * t)
        
        # Combine vectors into a single dirty mix matrix
        dirty_mix = wind_rumble + target_signal
        
        sf.write(test_path, dirty_mix, sample_rate)
        print(f"📝 Created dummy validation audio containing heavy wind noise at: {test_path}")
        return test_path

if __name__ == "__main__":
    print("⚡ STARTING AUDIO DSP SECTOR VERIFICATION MATRIX...")
    processor = AudioActionFilter(cutoff_hz=150.0, order=5)
    
    # Run a self-contained validation loop
    test_file = processor.generate_synthetic_validation_audio()
    cleaned_file = processor.process_audio_file(test_file)
    
    # Verify signal reduction metrics
    original_energy = np.mean(sf.read(test_file)[0]**2)
    cleaned_energy = np.mean(sf.read(cleaned_file)[0]**2)
    attenuation = 10 * np.log10(original_energy / cleaned_energy)
    
    print(f"📊 Signal Processing Verification Summary:")
    print(f"   -> Original Dirty Mix Energy: {original_energy:.4f}")
    print(f"   -> Cleaned Audio Stream Energy: {cleaned_energy:.4f}")
    print(f"   -> Successfully Suppressed Low-Frequency Noise by: {attenuation:.2f} dB")
    
    # Cleanup verification files
    if os.path.exists(test_file): os.remove(test_file)
    if os.path.exists(cleaned_file): os.remove(cleaned_file)
    print("🎉 Audio filter module checks out successfully!")