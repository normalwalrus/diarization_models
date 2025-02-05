from models.pyannote import PYANNOTE
from models.nemo import DiarInference
from models.reverb import REVERB

import os

EXAMPLE_AUDIO_PATH = 'examples/IS1002c.Mix-Headset.wav'
DIARIZATION_DATASET_PATH = 'ami_dataset/audio'
LIST_OF_AUDIO_FILES = [f for f in os.listdir(DIARIZATION_DATASET_PATH) if f.endswith(".wav")]

def main():

    pyannote_model = PYANNOTE()
    reverb_model = REVERB()
    nemo_model = DiarInference()
    count = 0

    for audio_file in LIST_OF_AUDIO_FILES:

        audio_filepath = DIARIZATION_DATASET_PATH + '/' + audio_file
        basename, _ = os.path.splitext(audio_file)
        print(f"Diarization for {audio_filepath} :")

        diarization = nemo_model.diarize_to_rttm(audio_filepath, f'outputs/nemo/{basename}.rttm')
        diarization = pyannote_model.diarize_into_rttm(audio_filepath, f'outputs/pyannote/{basename}.rttm')
        diarization = reverb_model.diarize_into_rttm(audio_file, f'outputs/reverb/{basename}.rttm')

        count += 1
        if count % 10 == 0:
            print('-----------------------'*6)
            print(f"We are at {count}!")
            print('-----------------------'*6)


if __name__ == '__main__':

    main()