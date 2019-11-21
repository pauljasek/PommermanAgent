FROM tensorflow/tensorflow:1.14.0-py3

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN git clone https://github.com/MultiAgentLearning/playground.git /playground
WORKDIR /playground
RUN pip install .

# Install TensorFlow and other dependencies
#RUN pip install tensorflow==1.9.0 dm-sonnet==1.23
#RUN pip install tensorflow==1.13.1 tensorflow-probability==0.6.0
RUN pip install dm-sonnet==1.23
ADD scalable_population /agent

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]

# Docker commands:
#   docker rm scalable_agent -v
#   docker build -t scalable_agent .
#   docker run --name scalable_agent scalable_agent
