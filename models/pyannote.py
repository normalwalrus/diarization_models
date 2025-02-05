import os
import torch
import logging

from pyannote.audio import Pipeline

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)

logger_nemo = logging.getLogger('nemo_logger')
logger_nemo.disabled = True

class PYANNOTE:

    def __init__(self):

        device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        logging.info("Running on device: %s", device)

        self.diarizer = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(self.device)
                #use_auth_token=os.environ['HF_TOKEN']).to(self.device)

        logging.info("Pyannote model loaded!")
        
    def diarize(self, audio_filepath):

        logging.info("Diarization started")
        diarization = self.diarizer(audio_filepath)
        simple_text = ''

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            print(f"start={turn.start:.3f}s stop={turn.end:.3f}s speaker_{speaker}")

        return