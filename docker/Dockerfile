FROM python:3.11-slim-bookworm AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_docker.txt ./
RUN python -m pip install --no-cache-dir -r requirements_docker.txt

COPY *.py ./
COPY templates/index.html ./templates/index.html

RUN groupadd --gid 10001 cmos \
    && useradd --uid 10001 --gid cmos --no-create-home cmos \
    && install -d -o cmos -g cmos /app/data

USER cmos
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "web_server:app", "--host", "0.0.0.0", "--port", "8000"]
