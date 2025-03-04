# docker build -t asr-eval/whisper-hf:1.0.0 .
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install libsndfile1 (linux soundfile package)
RUN apt-get clean \
    && apt-get update \ 
    && apt-get install -y gcc g++ libsndfile1 ffmpeg sox wget git \
    && rm -rf /var/lib/apt/lists/*

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1
ENV APP_ROOT=/opt/app-root\
    POETRY_VERSION=1.8.3

WORKDIR $APP_ROOT

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
RUN python3 -m pip install --upgrade --no-cache-dir pip wheel \ 
    && python3 -m pip install --no-cache-dir Cython==3.0.6

ADD requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

#ARG NEMO_VERSION=1.23.0
ARG NEMO_VERSION=2.1.0
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir Cython==0.29.35 && \
    pip3 install --no-cache-dir nemo_toolkit[asr]==${NEMO_VERSION}

#Used for initial_prompt
RUN ["python", "-c", "from nemo.collections.asr.models.msdd_models import NeuralDiarizer; NeuralDiarizer.from_pretrained('diar_msdd_telephonic')"]
RUN ["python", "-c", "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1',use_auth_token='HF_TOKEN')"]
RUN ["python", "-c", "from pyannote.audio import Pipeline; Pipeline.from_pretrained('Revai/reverb-diarization-v2',use_auth_token='HF_TOKEN')"]
RUN ["python", "-c", "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/voice-activity-detection',use_auth_token='HF_TOKEN')"]

ENTRYPOINT [ "bash" ]