FROM python:3.9

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgmp3-dev \
    libmpfr-dev \
    libmpc-dev \
    libssl-dev \
    libffi-dev \
    python-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*


RUN curl -L http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz | tar xz


WORKDIR /ta-lib
RUN ./configure --prefix=/usr \
    && make \
    && make install \
    && rm -rf /ta-lib


COPY requirements.txt /app/requirements.txt
COPY . /app


WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install TA-Lib

RUN pip install flask
RUN pip install flask_session

CMD ["python" , "app.py"]