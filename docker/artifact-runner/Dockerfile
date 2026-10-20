FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY runner.py /opt/idea-artifact-runner.py
WORKDIR /output
ENTRYPOINT ["python", "-I", "/opt/idea-artifact-runner.py"]
