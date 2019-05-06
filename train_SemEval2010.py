from load import load, TRAIN_FILENAME
import template_matcher
from elmo_re import TAGS
import utils.scorer as scorer

import torch
import torch.nn as nn
import torch.optim as optim
import random

TAGS = [
    'Other',
    'Cause-Effect(e1,e2)',
    'Cause-Effect(e2,e1)',
    'Component-Whole(e1,e2)',
    'Component-Whole(e2,e1)',
    'Content-Container(e1,e2)',
    'Content-Container(e2,e1)',
    'Entity-Destination(e1,e2)',
    'Entity-Destination(e2,e1)',
    'Entity-Origin(e1,e2)',
    'Entity-Origin(e2,e1)',
    'Instrument-Agency(e1,e2)',
    'Instrument-Agency(e2,e1)',
    'Member-Collection(e1,e2)',
    'Member-Collection(e2,e1)',
    'Message-Topic(e1,e2)',
    'Message-Topic(e2,e1)',
    'Product-Producer(e1,e2)',
    'Product-Producer(e2,e1)']

templates = [[tag, '<slot>'] for tag in TAGS]

def train():
    sents = load(TRAIN_FILENAME)

    test_sents = sents[:1000]
    test_tags = [s[2] for s in test_sents]

    shuffled_sents = sents[1000:].copy()

    net = template_matcher.TemplateMatcher()
    opt = optim.Adam(net.parameters())
    crit = nn.CrossEntropyLoss()
    BATCH_SIZE = 32
    epoch = 0

    try:
        checkpoint = torch.load('template_matcher_SemEval2010_latest.pt')
        net.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        epoch = checkpoint['epoch']
        print(f'Successfully resumed from epoch {epoch}.')
    except Exception as e:
        print(f'Could not resume training: {e}')

    while True:
        print(f'Evaluating epoch {epoch}')
        res_tags = []
        for i in range(len(test_sents) // 100):
            nl_ses = []
            for words, entities, target in test_sents[10 * i:10 * (i + 1)]:
                nl_s = []
                for word, entity in zip(words, entities):
                    nl_s.append(word)
                    if entity != 'null':
                        nl_s.append(entity)
                nl_ses.append(nl_s)
            encoded_template, slot_name = net.encode_templates(templates)
            slot_matches, temp_matches = net(encoded_template, nl_ses)
            res_tags.extend([TAGS[t] for t in temp_matches.argmax(dim=1)])
        scorer.evaluate(test_tags, res_tags)

        running_loss = 0.0
        running_accuracy = 0.0
        running_samples = 0.0

        random.shuffle(shuffled_sents)
        for i in range(len(shuffled_sents) // BATCH_SIZE - 1):
            if False:
                i = 0
            opt.zero_grad()
            ses = shuffled_sents[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            nl_ses = []
            ses[0]
            for words, entities, target in ses:
                nl_s = []
                for word, entity in zip(words, entities):
                    nl_s.append(word)
                    if entity != 'null':
                        nl_s.append(entity)
                nl_ses.append(nl_s)
            ses
            nl_ses

            encoded_template, slot_name = net.encode_templates(templates)
            slot_matches, temp_matches = net(encoded_template, nl_ses)
            tags = torch.tensor([TAGS.index(s[2]) for s in ses])
            loss = crit(temp_matches, tags)
            loss.backward()
            opt.step()
            running_loss += loss.item() / len(tags)
            res_long = torch.argmax(temp_matches, dim=1)
            accuracy = float((tags == res_long).sum()) / len(tags)
            running_accuracy = (running_samples * running_accuracy + len(tags)
                * accuracy) / (running_samples + len(tags))
            running_samples += len(tags)
            if i % 10 == 0 and i > 0:
                print(f'epoch {epoch}, iter {i} loss: {running_loss}, accuracy: {running_accuracy}')
                running_loss = 0.0
                running_accuracy = 0.0
                running_samples = 0
        checkpoint = {
            'model': net.state_dict(),
            'opt': opt.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, f'template_matcher_SemEval2010_{epoch}.pt')
        torch.save(checkpoint, 'template_matcher_SemEval2010_latest.pt')
        torch.save(net, f'template_matcher_SemEval2010_{epoch}.model')
        torch.save(net, f'template_matcher_SemEval2010_latest.model')
        print(f'Saved weights for epoch {epoch}')
        epoch += 1

if __name__ == '__main__':
    train()
