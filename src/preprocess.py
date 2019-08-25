import string


def preprocess(df):
    for lang in ["en", "fr"]:
        df[lang] = df[lang].str.lower()
        df[lang] = df[lang].str.replace(f'([{string.punctuation}])', r' \1 ')
        df[lang] = df[lang].str.replace(f'\\s+', r' ')
        df[lang] = df[lang].str.strip()
        df[lang] = df[lang].str.split()
    return df
