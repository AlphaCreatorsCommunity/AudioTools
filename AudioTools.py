import pydub
import librosa
import numpy as np
import pyrubberband as pyrb
import soundfile as sf
import tempfile

class AudioTools:
    sound_file = ""
    sound = None
    sound_sample = None
    sound_sample_sf = None

    def mp3towav(self, filepath, outfilepath):
        sound = pydub.AudioSegment.from_mp3(filepath)
        sound.export(outfilepath, format="wav")

    def shift(self, sample_width = 2, frame_rate = 44100, channels = 2, n_steps = 0):
        y_shift = pyrb.pitch_shift(y=self.sound_sample_sf, sr=frame_rate, n_steps=n_steps)

        tmpdir = tempfile.TemporaryDirectory()

        sf.write(tmpdir.name + "/out.wav", y_shift, frame_rate, format="wav")
        out = pydub.AudioSegment.from_wav(tmpdir.name + "/out.wav")

        tmpdir.cleanup()

        return out

    def calcTempo(self, frame_rate = 44100):
        tempo, _ = librosa.beat.beat_track(y=self.sound_sample.astype(np.float32), sr=frame_rate)
        
        return np.round(tempo)

    def hpss(self):
        harmonics, percussive = librosa.effects.hpss(y=self.sound_sample.astype(np.float32))

        return (harmonics, percussive)

    def makeAudioSegment(self, sample_width = 2, frame_rate = 44100, channels = 2):
        out = pydub.AudioSegment(self.sound_sample.astype("int16").tobytes(), sample_width=sample_width, frame_rate=frame_rate, channels=channels)

        return out
    
    def initDatas(self, sound_file: str):
        self.sound_file = sound_file
        self.sound = pydub.AudioSegment.from_wav(sound_file)
        self.sound_sample = np.array(self.sound.get_array_of_samples())
        self.sound_sample_sf = sf.read(sound_file)[0]