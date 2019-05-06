import os
import subprocess
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
    ['Edit', 'a', 'file', 'called', '<file>'],
    ['Create', 'a', 'file', 'called', '<file>'],
    ]

encoded_templates, slot_names = net.encode_templates(templates)

while True:
    try:
        print('engsh: ', end='', flush=True)
        user_input = input().strip().split(' ')
        slot_matches, template_matches = net(encoded_templates, [user_input])
        print('slot_matches', slot_matches)
        print('template_matches', template_matches)
        which_template = template_matches[0].argmax()
        match_likelihood = template_matches[0][which_template]
        matched_template = templates[which_template]
        # print(f'I think you said: {matched_template!r} with likelihood '
              # f'{match_likelihood!r}')
        slot_idx = slot_matches[0][0][0].argmax()
        if which_template == 0:
            dirname = user_input[slot_idx]
            os.chdir(dirname)
            print(f"I've entered the directory called {dirname!r}")
            print(f"(`cd {dirname!r}`)")
        elif which_template == 1:
            dirname = user_input[slot_idx]
            os.mkdir(dirname)
            print(f"I've created a directory called {dirname!r}")
            print(f"(`cd {dirname!r}`)")
        elif which_template == 2:
            dirname = user_input[slot_idx]
            if 'current' in dirname:
                print("(`ls`)")
                os.listdir()
            else:
                print(f"(`ls {dirname!r}`)")
                os.listdir(dirname)
        elif which_template == 3:
            filename = user_input[slot_idx]
            subprocess.call(['vim', filename])
        elif which_template == 4:
            filename = user_input[slot_idx]
            subprocess.call(['touch', filename])
            print(f"(`touch {filename!r}`)")
        elif which_template == 5:
            subprocess.call(['git', 'init'])
            print(f"(`git init`)")
        elif which_template == 6:
            filename = user_input[slot_idx]
            subprocess.call(['git', 'add', filename])
            print(f"(`git add {filename!r}`)")
        else:
            raise TypeError('Unknown template!')
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise e
        print(e)
