from models.pyannote import PYANNOTE
from models.nemo import DiarInference

EXAMPLE_AUDIO_PATH = 'examples/finance_zoom_meeting.wav'

def main():

    pyannote_model = PYANNOTE()
    nemo_model = DiarInference()

    diarization = nemo_model.diarize_to_rttm(EXAMPLE_AUDIO_PATH, 'outputs/nemo_test.rttm')
    print(diarization)

    diarization = pyannote_model.diarize_into_rttm(EXAMPLE_AUDIO_PATH, 'outputs/pyannote_test.rttm')
    print(diarization)


if __name__ == '__main__':

    main()