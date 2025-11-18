import os
import torch
import logging
import librosa
import torch
import numpy as np

from pyannote.audio import Pipeline, Model, Inference
from sklearn.cluster import AgglomerativeClustering

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO
)

logger_nemo = logging.getLogger('nemo_logger')
logger_nemo.disabled = True

MAX_VAD_DURATION = 300

class PYANNOTE_DECONSTRUCTED:

    def __init__(self):

        device ='cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        logging.info("Running on device: %s", device)

        self.VAD = Pipeline.from_pretrained("pyannote/voice-activity-detection").to(self.device)
                #use_auth_token=os.environ['HF_TOKEN']).to(self.device)
        self.embdding = Model.from_pretrained("pyannote/embedding").to(self.device)
        self.embedding_inference = Inference(self.embdding, window="whole")
        
        self.clustering = AgglomerativeClustering(n_clusters=None, 
                                                  distance_threshold=0.85, 
                                                  compute_full_tree = True,
                                                  metric="cosine",
                                                  linkage="average")

        logging.info("Pyannote Deconstructed Diarizer model loaded!")
        
    def read_audio_into_numpy(self, path):
        
        y, sr = librosa.load(path)
        
        return y, sr
        
    def get_VAD_segments(self, 
                         numpy_array_audio_whole, 
                         sr=16000, 
                         offset=0):
        
        audio_input = torch.from_numpy(numpy_array_audio_whole)
        audio_input = audio_input.unsqueeze(0)
        
        with torch.no_grad():
            output = self.VAD(
                {'waveform': audio_input, 'sample_rate': sr})
            
        output_list = []
        for speech in output._tracks.items():
            output_list.append([speech[0].start + offset, speech[0].end + offset])

        return output_list
    
    def get_embedding_from_segment(self, numpy_array):
        
        return self.embedding_inference(numpy_array)
    
    def get_array_of_clusters(self, array_of_embeddings):
        
        return self.clustering.fit(array_of_embeddings)
        
    
    def diarize(self, audio_filepath):
        
        logging.info("Diarizing Process started")
        
        numpy_array_audio_whole, sr = self.read_audio_into_numpy(audio_filepath)
        logging.info("VAD started")
        # NOTE: Seems like it only VADs 300sec max??
        # This is to split the Audio into 300sec splits if audio > 300secs
        if len(numpy_array_audio_whole)/sr > MAX_VAD_DURATION:
            number_of_splits = (len(numpy_array_audio_whole)/sr)%MAX_VAD_DURATION + 1
            split_arrays = np.array_split(numpy_array_audio_whole, number_of_splits)
            offset = 0
            list_of_segments = []
            
            for array in split_arrays:
                list_of_segments += self.get_VAD_segments(array, sr, offset)
                offset+=(len(array)/sr)
                
        else:
            list_of_segments = self.get_VAD_segments(numpy_array_audio_whole, sr)
            
        list_of_embeddings = []
        list_of_indexes_to_pop = []
        count = 0
        print(f"Number of Segments : {len(list_of_segments)}")
        
        logging.info("Embedding started")
        # Embedding for each segment of audio
        for x in range(len(list_of_segments)):
            segment = list_of_segments[x]
            
            start_frame, end_frame = int(segment[0]*sr), int(segment[1]*sr)
            numpy_array_audio_segment = numpy_array_audio_whole[start_frame:end_frame]
            
            if len(numpy_array_audio_segment) < sr:
                list_of_indexes_to_pop.append(x)
                count+=1
                continue
            
            embedding_segment = self.get_embedding_from_segment({"waveform":torch.from_numpy(numpy_array_audio_segment).unsqueeze(0), "sample_rate":sr})
            
            list_of_embeddings.append(embedding_segment)
            
        list_of_indexes_to_pop.sort(reverse=True)
        
        # Removing audio segments that are less than 1 sec
        for index in list_of_indexes_to_pop:
            list_of_segments.pop(index)

        print(f"Number of segments removed: {count}")
        
        # Clustering all Embeddings
        logging.info("Clustering started")
        clusters = self.get_array_of_clusters(list_of_embeddings)
        
        final_diarized_segments = []
        
        for y in range(len(list_of_segments)):
            
            current_speaker = clusters.labels_[y]
            current_segment = list_of_segments[y]
            
            final_diarized_segments.append([current_segment[0], current_segment[1], current_speaker])
            
        return final_diarized_segments
        
    def diarize_as_string(self, audio_filepath):
        
        final_diarized_segments = self.diarize(audio_filepath)
        final_string = ''
        
        for segment in final_diarized_segments:
            
            final_string+=f"start={segment[0]:.3f}s stop={segment[1]:.3f}s speaker_{segment[2]} \n"
            
        return final_string
                
    def diarize_as_rttm(self, audio_filepath, output_filepath = 'paynnote_deconstructed_output.rttm'):
        
        final_diarized_segments = self.diarize(audio_filepath)
        with open(output_filepath, "w") as rttm_file:
            
            for segment in final_diarized_segments:
                
                start_time = segment[0]
                end_time = segment[1]
                speaker = segment[2]
                duration = end_time - start_time
                
                rttm_line = f"SPEAKER {audio_filepath} 1 {start_time:.3f} {duration:.3f} <NA> <NA> speaker_{speaker} <NA>\n"
                rttm_file.write(rttm_line)
                
        return "Pyannote Deconstructed Done"
        