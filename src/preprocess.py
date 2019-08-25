import string


def preprocess(df):
    for lang in ["en", "fr"]:
        df[lang] = df[lang].str.lower()
        df[lang] = df[lang].str.replace(f'([{string.punctuation}])', r' \1 ')
        df[lang] = df[lang].str.replace(f'\\s+', r' ')
        df[lang] = df[lang].str.strip()
        df[lang] = df[lang].str.split()
    return df


class LanguageIndex:
    def __init__(self):
        self.index = {0: "<sos>", 1: "<eos>"}
        self.inverted_index = {token: i for i, token in self.index.items()}
        self.vocab_size = 2

    def add_sentences(self, sentences):
        for sentence in sentences:
            for token in sentence:
                if token not in self.inverted_index:
                    self.inverted_index[token] = self.vocab_size
                    self.index[self.vocab_size] = token
                    self.vocab_size += 1
