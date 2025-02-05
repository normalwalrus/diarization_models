from models.pyannote import PYANNOTE

EXAMPLE_AUDIO_PATH = 'examples/steroids_120sec.wav'

def main():

    pyannote_model = PYANNOTE()
    diarization = pyannote_model.diarize(EXAMPLE_AUDIO_PATH)

    print(diarization)


if __name__ == '__main__':

    main()