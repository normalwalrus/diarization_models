# Diarization_models

This is a repo to provide the tools to do diarisation inference and DER computation. 

List of supported models:

1. Pyannote Diariser v3.1
2. Pyannote community 1

## Setting up

1. Ensure you have a huggingface token and store it into a token.txt file. Refer to token_example.txt on how to store it.

2. Ensure you are volume mounting the correct dataset folder in the docker-compose.yaml, Example:
```yaml
volumes:
    - ./:/opt/app-root
    - /path/to/your/dataset/folder:/opt/app-root/data
```

3. Build the docker image
```bash
docker compose build
```

4. Run the docker image
```bash
docker compose run diarization_models
```

## Running the inference

1. In the pyannote_inference.py file, change these variables:

```python
if __name__ == '__main__':
    
    DIARIZATION_DATASET_PATH = 'data/PATH/TO/AUDIO/FOLDER'
    PYANNOTE_OUTPUT_FOLDER_PATH = 'PATH/TO/PYANNOTE/OUTPUT/FOLDER'
    PYANNOTE_COMMUNITY_OUTPUT_FOLDER_PATH = 'PATH/TO/PYANNOTE/COMMUNITY/OUTPUT/FOLDER'
```

2. Within the container, run:

```bash
python3 pyannote_inference.py
```

3. The .rttm files can be found at where you specified the output folder to be

## Compute the DER

1. In the compute_DER.py file, change these variables:
```python
if __name__ == "__main__":
    pred_dir = "PATH/TO/PYANNOTE/OUTPUT/FOLDER"
    gold_dir = "PATH/TO/GOLD/RTTM/FOLDER"
```

2. Within the container, run:

```bash
python3 compute_DER.py
```

## Acknowledgements

This repo was created and maintained by the hardworking and death DH employees