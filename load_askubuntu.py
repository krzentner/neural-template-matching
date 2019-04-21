def load_text(text_filename='askubuntu/text_tokenized.txt'):
    questions = {}
    with open(text_filename) as f:
        for line in f:
            identity, title, body = line.split('\t')
            questions[int(identity)] = title.strip().split(' ')
    return questions


def load_questions(filename='askubuntu/train_random.txt'):
    pairs = []
    with open(filename) as f:
        for line in f:
            base, dups, negatives = line.split('\t')
            base = int(base)
            for d in dups.split(' '):
                d = int(d)
                pairs.append((base, d, True))
            for n in negatives.split(' '):
                n = int(n)
                pairs.append((base, n, False))
    return pairs
