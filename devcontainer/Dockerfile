# Build Stage
FROM python:3.8-slim AS build

WORKDIR /Deer_Population_Prediction

COPY requirements.txt ./

# Install dependencies in a single layer for efficiency
RUN pip install --no-cache-dir -r requirements.txt


# Final Image
FROM python:3.8-slim

WORKDIR /Deer_Population_Prediction

# Copy the installed dependencies from the build stage
COPY --from=build /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages

# Copy the rest of the application
COPY . .

EXPOSE 5000

CMD ["bash"]


# docker build -t Deer_Population_Prediction
# docker run -it Deer_Population_Prediction
