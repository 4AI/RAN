DATA_DIR=data/booksummaries

if [ ! -d "$DATA_DIR" ]; then
    wget -P "$DATA_DIR" http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz
    tar -xf "${DATA_DIR}/booksummaries.tar.gz" -C data
fi

python booksummary_preprocess.py
