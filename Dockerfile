FROM ubuntu:22.04

WORKDIR /project 
COPY . .

RUN apt-get update 

RUN apt-get install python3-pip -y

RUN pip install -r requirements.txt
RUN pip install notebook
RUN python3 -m nltk.downloader stopwords wordnet punkt_tab

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
