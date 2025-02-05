import librosa
import soundfile as sf

def preprocess_audio(audio_filepath, output_path):

    y, sr = librosa.load(audio_filepath, sr=16000, mono=True)

    sf.write(output_path, y, sr)

if __name__ == '__main__':

    audio_filepath = 'C:\Users\User\Desktop\diarization_models\examples/finance_zoom_meeting.wav"
    output_path = 'C:\Users\User\Desktop\diarization_models\examples/finance_zoom_meeting_processed.wav"

    preprocess_audio(audio_filepath, output_path)