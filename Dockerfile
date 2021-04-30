FROM huggingface/transformers-pytorch-cpu:latest

WORKDIR /workspace

COPY . .

RUN ["pip","install","-r","./requirements.txt"]

EXPOSE 80

CMD ["python3","nli_api.py"]