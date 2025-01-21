FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

COPY src /workspace/src

CMD [ "python", "-u", "src/runpod_handler.py" ]