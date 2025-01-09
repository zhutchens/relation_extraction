FROM python:3.12.8-bookworm

WORKDIR /eduProject
COPY . .

RUN pip3 install --upgrade pip

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt
RUN pip3 install notebook
RUN python3 -m nltk.downloader wordnet stopwords punkt_tab

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
