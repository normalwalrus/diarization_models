from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.utils import nemo_logging
from typing import List, Union

import logging
import torch
import pandas as pd

logger_nemo = logging.getLogger('nemo_logger')
logger_nemo.disabled = True

class DiarInference:
    '''
    Diar inference class
    '''
    def __init__(self):

        device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.map_location = torch.device(device)

        self.diar_model = NeuralDiarizer.from_pretrained('diar_msdd_telephonic').to(self.map_location)

    def diarize(self, audio_path: str) -> pd.DataFrame:
        annotation = self.diar_model(audio_path, num_workers=0, batch_size=16)
        rttm=annotation.to_rttm()
        df = pd.DataFrame(columns=['start_time', 'end_time', 'speaker', 'text'])
        lines = rttm.splitlines()
        if len(lines) == 0:
            df.loc[0] = 0, 0, 'No speaker found'
            return df
        start_time, duration, prev_speaker = float(lines[0].split()[3]), float(lines[0].split()[4]), lines[0].split()[7]
        end_time = float(start_time) + float(duration)
        df.loc[0] = start_time, end_time, prev_speaker, ''

        for line in lines[1:]:
            split = line.split()
            start_time, duration, cur_speaker = float(split[3]), float(split[4]), split[7]
            end_time = float(start_time) + float(duration)
            if cur_speaker == prev_speaker:
                df.loc[df.index[-1], 'end_time'] = end_time
            else:
                df.loc[len(df)] = start_time, end_time, cur_speaker, ''
            prev_speaker = cur_speaker

        return df
    
    def diarize_to_rttm(self, audio_path:str, output_filepath:str):
        annotation = self.diarize(audio_path)

        with open(output_filepath, "w") as rttm_file:
            for _, row in annotation.iterrows():
                start_time = row["start_time"]
                duration = row["end_time"] - row["start_time"]
                speaker = row["speaker"]
                
                rttm_line = f"SPEAKER {audio_path} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker} <NA>\n"
                rttm_file.write(rttm_line)

        return "Nemo Diarization Done"

if __name__ == '__main__':
    
    if torch.cuda.is_available():
        DEVICE = [0]  # use 0th CUDA device
        ACCELERATOR = 'gpu'
    else:
        DEVICE = 1
        ACCELERATOR = 'cpu'
    
    model_name = 'diar_msdd_telephonic'
    path_to_example = 'example/steroids_120sec.wav'
    
    diar_model = DiarInference(model_name, DEVICE, ACCELERATOR)
    df = diar_model.diarize(path_to_example)
    
    print(df)