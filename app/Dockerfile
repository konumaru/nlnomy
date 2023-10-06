# Dockerfile_streamlit
FROM python:3.11.4

WORKDIR /app

RUN pip install -U pip 
RUN pip install poetry

COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false
RUN poetry install 
# RUN poetry export --output requirements.txt --without-hashes

COPY . .

CMD ["streamlit", "run", "main.py"]
