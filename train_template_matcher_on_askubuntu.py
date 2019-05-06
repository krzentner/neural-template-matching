import template_matcher
import load_askubuntu

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

import importlib
importlib.reload(template_matcher)

H = False

def train():
    question_titles = load_askubuntu.load_text()
    question_pairs = load_askubuntu.load_questions()

    shuffled_pairs = question_pairs.copy()
    random.shuffle(shuffled_pairs)
    holdout_pairs = shuffled_pairs[-1000:]
    train_pairs = shuffled_pairs[:-1000]

    net = template_matcher.TemplateMatcher()
    opt = optim.Adam(net.parameters())
    slot_crit = nn.CrossEntropyLoss()
    match_crit = nn.L1Loss()
    BATCH_SIZE = 32
    epoch = 0

    try:
        checkpoint = torch.load('template_matcher_askubuntu_latest.pt')
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

        def train_step(w_template, triples):
            nonlocal running_samples
            nonlocal running_loss
            nonlocal running_accuracy
            sentences = [triple[0] for triple in triples]
            slot_expectations = torch.tensor([triple[1] for triple in triples])
            match_expectations = [1. if triple[2] else -1. for triple in triples]

            # for index_in_b, w_b in enumerate(sentence):
            #     if w == w_b:
            #         slot_expectation = torch.tensor(index_in_b).reshape(1, 1, n_slots, 1)
            #         sentences.append(sentence)
            #         if should_match:
            #             match_expectations.append([1.])
            #         else:
            #             match_expectations.append([-1.])
            #         slot_expectations.append(slot_expectation)
            #
            # slot_expectations = torch.cat(slot_expectations, dim=0)

            opt.zero_grad()
            encoded_template, slot_name = net.encode_templates([w_template])
            slot_matches, temp_matches = net(encoded_template, sentences)
            # slot_matches has shape:
            # sentence, template, slot, word
            expected_match_tensor = torch.tensor(match_expectations)
            match_loss = match_crit(temp_matches, expected_match_tensor)

            slot_matches_flat = slot_matches.flatten(0, 2)
            slot_expectations_flat = slot_expectations
            loss = slot_crit(slot_matches_flat, slot_expectations_flat)
            best_slot_guess = slot_matches_flat.argmax()

            total_loss = 4 * match_loss + loss
            total_loss.backward()
            opt.step()

            accuracy = ((best_slot_guess == slot_expectations_flat).to(torch.float32)).mean()

            running_accuracy = (running_samples * running_accuracy + accuracy) / (running_samples + 1)
            running_loss = (running_samples * running_loss + total_loss) / (1 + running_samples)
            running_samples += 1

        for i, (first_question, second_question, should_match) in enumerate(train_pairs):
            if H:
                first_question, second_question, should_match = train_pairs[2]
            first_title = question_titles[first_question]
            second_title = question_titles[second_question]
            first_title
            second_title
            for a, b in [(first_title, second_title),
                         (second_title, first_title)]:
                if H:
                    a = first_title
                    b = second_title
                for index_in_a, w in enumerate(a):
                    if H:
                        index_in_a, w = next(enumerate(a))
                    w_slot = random.randrange(template_matcher.TEMPLATE_SLOT_SIZE)
                    w_template = list(a)
                    w_template[index_in_a] = f'<slot w[index_in_a]>'
                    triples = []
                    for index, wb in enumerate(b):
                        if H:
                            index, wb = next(enumerate(b))
                        if wb == w and len(w) > 2:
                            # Cross-example match!
                            triples.append((b, index, should_match))
                        else:
                            if index >= len(a):
                                continue
                            s = list(a)
                            if random.getrandbits(1) or index == index_in_a:
                                # Clobber the word from a with one from b.
                                s[index] = wb
                                triples.append((s, index_in_a, True))
                            else:
                                # Delete the word at `index`.
                                del s[index]
                                if index > index_in_a:
                                    triples.append((s, index_in_a, True))
                                else:
                                    triples.append((s, index_in_a - 1, True))
                        # Balance out match and not-match by using (not matching) shuffled input.
                        # Note: since bag-of-words should sort of work, this is probably a bad method.
                        shuffling = list(np.random.permutation(len(a)))
                        a_shuffled = [a[shuffle] for shuffle in shuffling]
                        shuffled_index = shuffling.index(index_in_a)
                        triples.append((a_shuffled, shuffled_index, False))
                    train_step(w_template, triples)


            if i % 10 == 0 and i > 0:
                print(f'epoch {epoch}, iter {i} loss: {running_loss}, accuracy: {running_accuracy}')
                running_loss = 0.0
                running_accuracy = 0.0
                running_samples = 0
            if i % 100 == 0 and i > 0:
                print('Saving checkpoint mid epoch')
                checkpoint = {
                    'model': net.state_dict(),
                    'opt': opt.state_dict(),
                    'epoch': epoch,
                }
                torch.save(checkpoint, 'template_matcher_askubuntu_latest.pt')
                torch.save(net, f'template_matcher_askubuntu_latest.model')
        checkpoint = {
            'model': net.state_dict(),
            'opt': opt.state_dict(),
            'epoch': epoch,
        }
        torch.save(checkpoint, f'template_matcher_askubuntu_{epoch}.pt')
        torch.save(checkpoint, 'template_matcher_askubuntu_latest.pt')
        torch.save(net, f'template_matcher_askubuntu_{epoch}.model')
        torch.save(net, f'template_matcher_askubuntu_latest.model')
        print(f'Saved weights for epoch {epoch}')
        epoch += 1

if __name__ == '__main__':
    train()
