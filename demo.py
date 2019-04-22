import os
import template_matcher
import torch

try:
    net = torch.load('template_matcher_askubuntu_latest.model')
except Exception as e:
    print('Could not load template matcher.')
    raise e

templates = [
    ['Go', 'to', '<dir>'],
    ['Make', 'a', 'directory', 'called', '<dir>'],
    ['List', '<dir>'],
    ]

encoded_templates = net.encode_templates(templates)

while True:
    print('engsh: ', end='', flush=True)
    user_input = input().strip().split(' ')
    slot_matches, template_matches = net(encoded_templates, [user_input])
    which_template = template_matches[0].argmax()
    match_likelihood = template_matches[0][which_template]
    matched_template = templates[which_template]
    print(f'I think you said: {matched_template!r} with likelihood {match_likelihood!r}')
    slot_idx = slot_matches[0][0][0].argmax()
    if which_template == 0:
        dirname = user_input[slot_idx]
        os.chdir(dirname)
    elif which_template == 1:
        dirname = user_input[slot_idx]
        os.mkdir(dirname)
    else:
        dirname = user_input[slot_idx]
        if 'current' in dirname:
            os.listdir()
        else:
            os.listdir(dirname)
