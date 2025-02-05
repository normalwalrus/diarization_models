import os
import torch
import logging

from pyannote.audio import Pipeline

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)

logger_nemo = logging.getLogger('nemo_logger')
logger_nemo.disabled = True

class REVERB:

    def __init__(self):

        device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        logging.info("Running on device: %s", device)

        self.diarizer = Pipeline.from_pretrained("Revai/reverb-diarization-v2").to(self.device)
                #use_auth_token=os.environ['HF_TOKEN']).to(self.device)

        logging.info("Revarb model loaded!")
        
    def diarize_into_string(self, audio_filepath):

        logging.info("Diarization started")
        diarization = self.diarizer(audio_filepath)
        simple_text = ''

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            simple_text += f"start={turn.start:.3f}s stop={turn.end:.3f}s speaker_{speaker} \n"

        return simple_text
    
    def diarize_into_rttm(self, audio_filepath, output_filepath):

        logging.info("Diarization started")
        diarization = self.diarizer(audio_filepath)

        with open(output_filepath, "w") as rttm:
            diarization.write_rttm(rttm)

        return 'Revarb Diarization Done'