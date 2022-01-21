from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data("train")

assert len(word_to_id) == len(id_to_word)
print(type(id_to_word))