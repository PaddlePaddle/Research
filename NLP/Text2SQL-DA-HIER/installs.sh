pip install torch==1.6.0
pip install jsonnet==0.14.0
pip install setproctitle
pip install tqdm==4.36.1
pip install asdl==0.1.5
pip install astor==0.7.1
pip install networkx==2.2
pip install pyrsistent
pip install sentencepiece
pip install attrs==18.2.0
pip install babel==2.7.0
pip install nltk==3.4
pip install bpemb==0.2.11
pip install cython==0.29.1
pip install transformers==2.3.0
pip install tabulate==0.8.6
pip install pytest==5.3.2
pip install stanford-corenlp==3.9.2
pip install entmax
pip install torchtext==0.7.0
pip install records==0.5.3

python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

cd parsers/RAT-SQL/
mkdir -p third_party && \
    cd third_party && \
    curl https://download.cs.stanford.edu/nlp/software/stanford-corenlp-full-2018-10-05.zip | jar xv
