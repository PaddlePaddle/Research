mkdir -p bert-large-uncased && cd bert-large-uncased &&
    wget https://huggingface.co/bert-large-uncased/raw/main/config.json &&
    wget https://huggingface.co/bert-large-uncased/raw/main/vocab.txt &&
    wget https://huggingface.co/bert-large-uncased/resolve/main/pytorch_model.bin &&
    cd ..