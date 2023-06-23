import os
import json
import pandas as pd


base_dir = 'data/booksummaries'


def parse_json_column(genre_data):
    """
    Read genre information as a json string and convert it to a dict
    :param genre_data: genre data to be converted
    :return: dict of genre names
    """
    try:
        return json.loads(genre_data)
    except Exception as e:
        return None # when genre information is missing


def load_booksummaries_data(book_path):
    """
    Load the Book Summary data and split it into train/dev/test sets
    :param book_path: path to the booksummaries.txt file
    :return: train, dev, test as pandas data frames
    """
    book_df = pd.read_csv(book_path, sep='\t', names=["Wikipedia article ID",
                                                      "Freebase ID",
                                                      "Book title",
                                                      "Author",
                                                      "Publication date",
                                                      "genres",
                                                      "summary"],
                          converters={'genres': parse_json_column})
    book_df = book_df.dropna(subset=['genres', 'summary']) # remove rows missing any genres or summaries
    book_df['word_count'] = book_df['summary'].str.split().str.len()
    book_df = book_df[book_df['word_count'] >= 10]
    train = book_df.sample(frac=0.8, random_state=22)
    rest = book_df.drop(train.index)
    dev = rest.sample(frac=0.5, random_state=22)
    test = rest.drop(dev.index)
    return train, dev, test


def prepare_book_summaries(pairs=False, book_path='data/booksummaries/booksummaries.txt'):
    """
    Load the Book Summary data and prepare the datasets
    :param pairs: whether to combine pairs of documents or not
    :param book_path: path to the booksummaries.txt file
    :return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(book_path):
        raise Exception("Data not found: {}".format(book_path))

    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}
    train, dev, test = load_booksummaries_data(book_path)

    if not pairs:
        print('train:', train)
        text_set['train'] = train['summary'].tolist()
        text_set['dev'] = dev['summary'].tolist()
        text_set['test'] = test['summary'].tolist()

        train_genres = train['genres'].tolist()
        label_set['train'] = [list(genre.values()) for genre in train_genres]
        dev_genres = dev['genres'].tolist()
        label_set['dev'] = [list(genre.values()) for genre in dev_genres]
        test_genres = test['genres'].tolist()
        label_set['test'] = [list(genre.values()) for genre in test_genres]
    else:
        train_temp = train['summary'].tolist()
        dev_temp = dev['summary'].tolist()
        test_temp = test['summary'].tolist()

        train_genres = train['genres'].tolist()
        train_genres_temp = [list(genre.values()) for genre in train_genres]
        dev_genres = dev['genres'].tolist()
        dev_genres_temp = [list(genre.values()) for genre in dev_genres]
        test_genres = test['genres'].tolist()
        test_genres_temp = [list(genre.values()) for genre in test_genres]

        for i in range(0, len(train_temp) - 1, 2):
            text_set['train'].append(train_temp[i] + train_temp[i+1])
            label_set['train'].append(list(set(train_genres_temp[i] + train_genres_temp[i+1])))

        for i in range(0, len(dev_temp) - 1, 2):
            text_set['dev'].append(dev_temp[i] + dev_temp[i+1])
            label_set['dev'].append(list(set(dev_genres_temp[i] + dev_genres_temp[i+1])))

        for i in range(0, len(test_temp) - 1, 2):
            text_set['test'].append(test_temp[i] + test_temp[i+1])
            label_set['test'].append(list(set(test_genres_temp[i] + test_genres_temp[i+1])))

    return text_set, label_set 


text_set, label_set = prepare_book_summaries()

for split in ['train', 'dev', 'test']:
    texts = text_set[split]
    labels = label_set[split]
    assert len(texts) == len(labels)
    print(f'{split} size:', len(texts))
    with open(os.path.join(base_dir, f'{split}.jsonl'), 'w') as writer:
        for text, label in zip(texts, labels):
            writer.writelines(json.dumps({'text': text, 'label': label}, ensure_ascii=False) + '\n')
