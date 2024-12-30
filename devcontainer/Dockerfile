FROM python:3.8-slim

WORKDIR /Deer_Population_Prediction

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /Deer_Population_Prediction/

CMD ["bash"]

# docker build -t Deer_Population_Prediction
# docker run Deer_Population_Prediction
