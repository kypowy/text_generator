import regex as re
import glob
import torch
from ast import literal_eval


def tokenize(text: str, batch_size: int, stride: int, word_freq: int):
    """generate tokenized text and dict

    Args:
        text (str): row text
        batch_size (int): size of batch
        stride (int): lenth of window in Strided Segmentation
        word_freq (int): frequency of words in text (from, to)

    Returns:
        (list):
            - source (list[list]): tokenized, devided by batches text
            - learn_list (list): list of unique words
    """
    clear_text = re.findall(r"\b\p{L}+\b", text.lower())
    learn_list = {}

    for word in clear_text:  # уникальные слова
        try:
            if word:
                learn_list[word] += 1

        except KeyError:
            learn_list[word] = 1

    unique_words = [word for word, freq in learn_list.items()
                    if word_freq[1] >= freq >= word_freq[0]]

    index_dict = {word: num for num, word in enumerate(unique_words)}

    # batch sigmentation
    word_indices = [index_dict[word] for word in clear_text
                    if word in unique_words]

    elements = [word_indices[i: i + batch_size]
                for i in range(0, len(word_indices), stride)]

    source = [elements[i: i + batch_size + 1]
              for i in range(0, len(elements), batch_size)]

    return source, unique_words


def check_dataset(text: str):
    """generate info about text (counts of words)

    Args:
        text (str): text for data extraction

    Returns:
        (dict): data info
    """
    clear_text = re.sub(r"[^\pL ]", " ", text)
    clear_text = clear_text.lower().split(" ")
    data_info = {}

    for word in clear_text:
        try:
            if word:
                data_info[word] += 1

        except KeyError:
            data_info[word] = 1

    return list(data_info.items())[: 100]


TEXT_LENTH = 30
STRIDE = 20
FREQUENCY = (2, 700)

with open("src_folder/source.txt", "r", encoding="utf-8") as src_file, \
     open("src_folder/dictionary.txt", "r", encoding="utf-8") as dict_file:
    src = literal_eval(src_file.read())
    learn_dict = literal_eval(dict_file.read())


if __name__ == "__main__":
    learning_text = str()

    for file_path in glob.glob("./texts" + "**/*.txt")[: 2000]:
        with open(file_path, mode="r", encoding="UTF-8") as file:
            learning_text += " " + file.read()

    src, learn_dict = tokenize(learning_text, TEXT_LENTH, STRIDE, FREQUENCY)

    print(torch.tensor(src[: -1]).size(), len(learn_dict))
    print([learn_dict[item] for item in src[0][1]])
    print(check_dataset(learning_text))

    with open("src_folder/source.txt", "w", encoding="utf-8") as data_file:
        data_file.write(str(src))

    with open("src_folder/dictionary.txt", "w", encoding="utf-8") as data_file:
        data_file.write(str(learn_dict))
