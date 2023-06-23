DATA_DIR=data/EURLEX57K

if [ ! -d "$DATA_DIR" ]; then
    echo "downloading eurlex57k..."
    mkdir "$DATA_DIR"
    wget -O "${DATA_DIR}/datasets.zip" http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
    unzip "${DATA_DIR}/datasets.zip" -d "$DATA_DIR"
    rm "${DATA_DIR}/datasets.zip"
    rm -rf "${DATA_DIR}/__MACOSX"
    mv "${DATA_DIR}/dataset/*" "$DATA_DIR"
    python eurlex_preprocess.py
    rm -rf "${DATA_DIR}/dataset"
    wget -O "${DATA_DIR}/EURLEX57K.json" http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json
fi
