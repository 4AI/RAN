DATA_DIR=data/hyperpartisan

if [ ! -d "$DATA_DIR" ]; then
    echo "downloading hyperpartisan to data/hyperpartisan"
    mkdir "$DATA_DIR"
    wget -P "$DATA_DIR" https://zenodo.org/record/1489920/files/articles-training-byarticle-20181122.zip
    wget -P "$DATA_DIR" https://zenodo.org/record/1489920/files/ground-truth-training-byarticle-20181122.zip
    unzip "${DATA_DIR}/articles-training-byarticle-20181122.zip" -d "$DATA_DIR"
    unzip "${DATA_DIR}/ground-truth-training-byarticle-20181122.zip" -d "$DATA_DIR"
    rm "${DATA_DIR}/*zip"
fi

if [ ! -e "${DATA_DIR}/hp-splits.json" ]; then
    echo "download hp-splits.json..."
    wget https://raw.githubusercontent.com/allenai/longformer/master/scripts/hp-splits.json -O "${DATA_DIR}/hp-splits.json"
fi
if [ ! -e "${DATA_DIR}/hp_preprocess.py" ]; then
    echo "download hp_preprocess.py..."
    wget https://raw.githubusercontent.com/allenai/longformer/master/scripts/hp_preprocess.py -O "${DATA_DIR}/hp_preprocess.py"
fi
python "${DATA_DIR}/hp_preprocess.py" --train-file "${DATA_DIR}/articles-training-byarticle-20181122.xml" --labels-file "${DATA_DIR}/ground-truth-training-byarticle-20181122.xml" --splits-file "${DATA_DIR}/hp-splits.json" --output-dir "$DATA_DIR"
