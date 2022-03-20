<<<<<<< HEAD
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY ./app /app

COPY ./models /models

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]
=======
FROM jupyter/scipy-notebook:0ce64578df46

RUN pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchtext==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/work"

RUN echo "export PYTHONPATH=/home/jovyan/work" >> ~/.bashrc

WORKDIR /home/jovyan/work

>>>>>>> master
