from models.pyannote_community import PYANNOTE_COMMUNITY
from models.pyannote import PYANNOTE

import os

import logging
logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)

def pyannote_inference_loop(
    diarisation_dataset_folder_path:str,
    output_folder_path:str,
    model
):
    os.makedirs(output_folder_path, exist_ok=True)
    pyannote_model = model

    rttm_files_done = [f for f in os.listdir(output_folder_path) if f.endswith(".rttm")]
    rttm_basename_done_list = []
    for rttm_file in rttm_files_done:
        rttm_basename_done, _ = os.path.splitext(rttm_file)
        rttm_basename_done_list.append(rttm_basename_done)

    count = 0
    list_of_audio_files = [f for f in os.listdir(diarisation_dataset_folder_path) if f.endswith(".wav")]

    for audio_file in list_of_audio_files:

        audio_filepath = diarisation_dataset_folder_path + '/' + audio_file
        basename, _ = os.path.splitext(audio_file)

        if basename in rttm_basename_done_list:
            print(f"Skipped {audio_file}: Already diarized")
            continue

        print(f"Diarization for {audio_filepath} :")

        diarization = pyannote_model.diarize_into_rttm(audio_filepath, os.path.join( output_folder_path ,f"{basename}.rttm"))

        count += 1
        if count % 10 == 0:
            print('-----------------------'*6)
            print(f"We are at {count}!")
            print('-----------------------'*6)


if __name__ == '__main__':
    
    for x in [2,3,4,5,6,7,8]:
        DIARIZATION_DATASET_PATH = f'data//NSC_PART4_OVERLAP_DIARISATION/{x}_speakers/audio'
        PYANNOTE_COMMUNITY_OUTPUT_FOLDER_PATH = 'outputs/pyannote_community_1/NSC4_overlap'
        PYANNOTE_DIARISER_OUTPUT_FOLDER_PATH = 'outputs/pyannote_diariser/NSC4_overlap'
        
        pyannote_model = PYANNOTE()
        pyannote_community_model = PYANNOTE_COMMUNITY()
        
        logging.info("Pyannote Diariser v3.1 inference started!")
        pyannote_inference_loop(
            diarisation_dataset_folder_path=DIARIZATION_DATASET_PATH,
            output_folder_path=PYANNOTE_DIARISER_OUTPUT_FOLDER_PATH,
            model=pyannote_model
        )
        
        logging.info("Pyannote Community 1 inference started!")
        pyannote_inference_loop(
            diarisation_dataset_folder_path=DIARIZATION_DATASET_PATH,
            output_folder_path=PYANNOTE_COMMUNITY_OUTPUT_FOLDER_PATH,
            model=pyannote_community_model
        )