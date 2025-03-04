from models.nemo import DiarInference

nemo_model = DiarInference()
path = 'examples/steroids_120sec.wav'

print(nemo_model.diarize(path))