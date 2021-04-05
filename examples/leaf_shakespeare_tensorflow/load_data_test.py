import load_data


if __name__ == "__main__":
    print(load_data.ALL_LETTERS)
    print(load_data.letter_to_vec("a"))
    print(load_data.word_to_indices("abc"))

    xy_train, xy_test = load_data.load_data("shakespeare/data/train", "shakespeare/data/test", 0)
