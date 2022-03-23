FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt .

RUN pip install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install -r requirements.txt

COPY ./app /app

COPY ./models /models

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]


# not required for Heroku - these were for other lab exercises
# torchvision==0.10.0+cpu torchtext==0.10.0
