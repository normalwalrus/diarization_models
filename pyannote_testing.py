from models.pyannote import PYANNOTE
from models.pyannote_VAD import PYANNOTE_VAD
<<<<<<< HEAD
from models.pyannote_deconstructed import PYANNOTE_DECONSTRUCTED
import time

# pyannote_model = PYANNOTE()
# pyannote_VAD_model = PYANNOTE_VAD()
pyannote_deconstructed = PYANNOTE_DECONSTRUCTED()
=======
import time

pyannote_model = PYANNOTE()
pyannote_VAD_model = PYANNOTE_VAD()
>>>>>>> 7b461262554b0472a400c1deaeeaf645476624a2

path = 'examples/ES2005a.Mix-Headset.wav'


def main():
    
<<<<<<< HEAD
    # pyannote_diarizer_start_time = time.time()
    
    # pyannote_model.diarize_into_string(path)
    
    # print(f"Pyannote Diarizer completed in {time.time() - pyannote_diarizer_start_time} seconds")
    
    # pyannote_VAD_start_time = time.time()
    
    # pyannote_VAD_model.VAD_into_string(path)
    
    # print(f"Pyannote VAD completed in {time.time() - pyannote_VAD_start_time} seconds")
    
    pyannote_deconsturcted_start_time = time.time()
    
    pyannote_deconstructed.diarize_as_rttm(path, output_filepath="testing.rttm")
    
    print(f"Pyannote Deconstructed completed in {time.time() - pyannote_deconsturcted_start_time} seconds")
=======
    pyannote_diarizer_start_time = time.time()
    
    pyannote_model.diarize_into_string(path)
    
    print(f"Pyannote Diarizer completed in {time.time() - pyannote_diarizer_start_time} seconds")
    
    pyannote_VAD_start_time = time.time()
    
    pyannote_VAD_model.VAD_into_string(path)
    
    print(f"Pyannote VAD completed in {time.time() - pyannote_VAD_start_time} seconds")
    
>>>>>>> 7b461262554b0472a400c1deaeeaf645476624a2
    
if __name__ == "__main__":
    main()
