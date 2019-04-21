import template_matcher
import load_askubuntu

import torch
import torch.nn as nn
import torch.optim as optim
import random

import importlib
importlib.reload(template_matcher)

def train():
    question_titles = load_askubuntu.load_text()
    question_pairs = load_askubuntu.load_questions()

    shuffled_pairs = question_pairs.copy()
    random.shuffle(shuffled_pairs)
    holdout_pairs = shuffled_pairs[-1000:]
    train_pairs = shuffled_pairs[:-1000]

    net = template_matcher.TemplateMatcher()
    opt = optim.Adam(net.parameters())
    crit = nn.CrossEntropyLoss()
    BATCH_SIZE = 32
    epoch = 0

    try:
        checkpoint = torch.load('elmo_ner_latest.pt')
        net.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        epoch = checkpoint['epoch']
        print(f'Successfully resumed from epoch {epoch}.')
    except Exception as e:
        print(f'Could not resume training: {e}')

    while True:
        running_loss = 0.0
        running_accuracy = 0.0
        running_samples = 0.0

        random.shuffle(train_pairs)
        for first_question, second_question, should_match in train_pairs:

            # first_question, second_question, should_match = next(iter(train_pairs))

            first_title = question_titles[first_question]
            second_title = question_titles[second_question]
            first_title
            second_title
            for a, b in [(first_title, second_title),
                         (second_title, first_title)]:
                # a = first_title
                # b = second_title
                for i, w in enumerate(a):
                    if len(w) > 2 and w in b:
                        w_slot = random.randrange(template_matcher.TEMPLATE_SLOT_SIZE)
                        w_template = list(a)
                        w_template[i] = f'<slot {w_slot}>'

                        sentences = []
                        match_expectations = []
                        slot_expectations = []
                        for index_in_b, w_b in enumerate(b):
                            if w == w_b:
                                slot_expectation = torch.zeros(len(b),
                                                               template_matcher.TEMPLATE_SLOT_SIZE)
                                slot_expectation[index_in_b, w_slot] = 1.
                                sentences.append(b)
                                slot_expectations.append(slot_expectation)
                                match_expectations.append(int(should_match))

                        opt.zero_grad()
                        encoded_template = net.encode_templates([w_template])
                        print(encoded_template.shape)
                        template_matches, slot_matches = net(encoded_template, sentences)
                        print(template_matches.shape)
                        print(slot_matches.shape)

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
        torch.save(checkpoint, f'elmo_ner_{epoch}.pt')
        torch.save(checkpoint, 'elmo_ner_latest.pt')
        torch.save(net, f'elmo_ner_{epoch}.model')
        torch.save(net, f'elmo_ner_latest.model')
        print(f'Saved weights for epoch {epoch}')
        epoch += 1

if __name__ == '__main__':
    train()
