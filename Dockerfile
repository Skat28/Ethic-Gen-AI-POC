FROM ubuntu:20.04
RUN apt update && apt install -y software-properties-common 
RUN add-apt-repository ppa:deadsnakes/ppa && apt update
RUN apt install -y python3.10 \
    python3.10-dev \
    git-all \
    unzip \
    curl
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python
WORKDIR /home/ubuntu
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN git clone https://github.com/Skat28/Ethic-Gen-AI-POC.git
WORKDIR ./Ethic-Gen-AI-POC
RUN mkdir pdf-files
RUN mkdir output-files
#aws installation:
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install
RUN apt install -y python3.10-venv
RUN python -m venv env
RUN /bin/bash -c "source env/bin/activate"
RUN pip3.10 install -r requirement.txt
RUN set -o allexport; source ".env"; set +o allexport
ENV OPENAI_API_KEY="${OPENAI_API_KEY}"
ENV PINECONE_API_KEY="${PINECONE_API_KEY}"
ENV AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
ENV AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
#ENV AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}
