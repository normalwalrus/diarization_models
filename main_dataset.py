from models.pyannote import PYANNOTE
from models.nemo import DiarInference
from models.reverb import REVERB
from models.pyannote_deconstructed import PYANNOTE_DECONSTRUCTED

import os

EXAMPLE_AUDIO_PATH = 'examples/IS1002c.Mix-Headset.wav'
DIARIZATION_DATASET_PATH = 'ami_dataset/audio'
LIST_OF_AUDIO_FILES = [f for f in os.listdir(DIARIZATION_DATASET_PATH) if f.endswith(".wav")]

def main():

    # pyannote_model = PYANNOTE()
    # reverb_model = REVERB()
    # nemo_model = DiarInference()
    pyannote_deconstructed_model = PYANNOTE_DECONSTRUCTED()

    rttm_files_done = [f for f in os.listdir('outputs/pyannote_deconstructed') if f.endswith(".rttm")]
    rttm_basename_done_list = []
    for rttm_file in rttm_files_done:
        rttm_basename_done, _ = os.path.splitext(rttm_file)
        rttm_basename_done_list.append(rttm_basename_done)
    
    # NOTE: To be used only to let nemo catch up        
    # nemo_rttm_files_done = [f for f in os.listdir('outputs/nemo') if f.endswith(".rttm")]
    # nemo_rttm_basename_done_list = []
    # for rttm_file in nemo_rttm_files_done:
    #     rttm_basename_done, _ = os.path.splitext(rttm_file)
    #     nemo_rttm_basename_done_list.append(rttm_basename_done)

    count = 0

    for audio_file in LIST_OF_AUDIO_FILES:

        # Prevent duplicate diarization (Assuming reverb is the last diarizer to finish)
        audio_filepath = DIARIZATION_DATASET_PATH + '/' + audio_file
        basename, _ = os.path.splitext(audio_file)

        if basename in rttm_basename_done_list:
            print(f"Skipped {audio_file}: Already diarized")
            continue

        print(f"Diarization for {audio_filepath} :")

        # diarization = nemo_model.diarize_to_rttm(audio_filepath, f'outputs/nemo/{basename}.rttm')
        # diarization = pyannote_model.diarize_into_rttm(audio_filepath, f'outputs/pyannote/{basename}.rttm')
        # diarization = reverb_model.diarize_into_rttm(audio_filepath, f'outputs/reverb/{basename}.rttm')
        diarization = pyannote_deconstructed_model.diarize_as_rttm(audio_filepath, f'outputs/pyannote_deconstructed/{basename}.rttm')

        count += 1
        if count % 10 == 0:
            print('-----------------------'*6)
            print(f"We are at {count}!")
            print('-----------------------'*6)


if __name__ == '__main__':

    main()