import os
import torch
import logging

from pyannote.audio import Pipeline

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)

logger_nemo = logging.getLogger('nemo_logger')
logger_nemo.disabled = True

class PYANNOTE_VAD:

    def __init__(self):

        device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        logging.info("Running on device: %s", device)

        self.VAD = Pipeline.from_pretrained("pyannote/voice-activity-detection").to(self.device)
                #use_auth_token=os.environ['HF_TOKEN']).to(self.device)

        logging.info("Pyannote VAD model loaded!")
        
    def VAD_into_string(self, audio_filepath):

        logging.info("VAD started")
        output = self.VAD(audio_filepath)
        simple_text = ''

        for speech in output.get_timeline().support():
            simple_text += f"start={speech.start:.3f}s stop={speech.end:.3f}s\n"

        return simple_text
    