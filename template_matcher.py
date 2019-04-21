import re

import torch
import torch.nn as nn

from allennlp.modules.elmo import Elmo, batch_to_ids

OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

ELMO_SIZE = 2048
HIDDEN_SIZE = 512
TEMPLATE_ENCODING_SIZE = 512
TEMPLATE_ENCODING_SIZE_DIR = int(TEMPLATE_ENCODING_SIZE / 2)
TEMPLATE_SLOT_SIZE = 16

SLOT_RE = re.compile(r'\s*\<.*?([0-9]+)\s*\>\s*')

def bidirectional_rnn_output_edges(output):
        output_with_dirs = output.view(output.shape[0], output.shape[1], 2, -1)
        last_forward_output = output_with_dirs[:, -1, 0]
        last_backward_output = output_with_dirs[:, 0, 1]
        repr = torch.cat([last_forward_output, last_backward_output], dim=1)
        return repr


class TemplateMatcher(nn.Module):
    def __init__(self):
        super(TemplateMatcher, self).__init__()
        # Note that we only need to download these weights during training.
        # However, there doesn't appear to be any way of initializing the
        # Elmo Module without these parameters.
        # Fortunately, we can skip this by pickling the whole model.
        self.elmo = Elmo(OPTIONS_FILE, WEIGHT_FILE, 2, dropout=0)
        self.template_encoder = nn.GRU(ELMO_SIZE + TEMPLATE_SLOT_SIZE,
                                       TEMPLATE_ENCODING_SIZE_DIR,
                                       num_layers=2, batch_first=True,
                                       bidirectional=True)
        self.encoder = nn.GRU(ELMO_SIZE + TEMPLATE_ENCODING_SIZE, HIDDEN_SIZE,
                              num_layers=2, batch_first=True,
                              bidirectional=True)
        self.slot_match_predictor = nn.Linear(2 * HIDDEN_SIZE,
                                              TEMPLATE_SLOT_SIZE)
        self.template_match_predictor = nn.Linear(2 * HIDDEN_SIZE, 1)

    def encode_templates(self, templates):
        embedded = self.elmo(batch_to_ids(templates))
        print(embedded)
        word_repr = torch.cat(embedded['elmo_representations'],
                                         dim=2)
        slot_repr = torch.zeros(word_repr.shape[0], word_repr.shape[1],
                                TEMPLATE_SLOT_SIZE)
        mask = torch.ones(word_repr.shape[0], word_repr.shape[1], 1)
        for i, t in enumerate(templates):
            for j, w in enumerate(t):
                match = SLOT_RE.match(w)
                if match is not None:
                    try:
                        slot_number = int(match.group(1))
                        assert slot_number < TEMPLATE_SLOT_SIZE
                    except:
                        raise TypeError(f'Invalid template slot {w!r}')
                    slot_repr[i, j, slot_number] = 1.
                    mask[i, j] = 0.
                elif '<' in w or '>' in w:
                    raise TypeError(f'Invalid template slot {w!r}')
        masked_repr = mask * word_repr
        template_pre_encoding = torch.cat((masked_repr, slot_repr), dim=2)
        output, h = self.template_encoder(template_pre_encoding)
        template_repr = bidirectional_rnn_output_edges(output)
        return template_repr

    def forward(self, templates, sentences):
        embedded = self.elmo(batch_to_ids(sentences))
        word_repr = torch.cat(embedded['elmo_representations'], dim=2)
        # Add a second (outer) batch dimension, so that we can concatenate in
        # the sentence representations.
        per_sent_temp = templates.unsqueeze(0).expand(word_repr.shape[0],
                                                      *templates.shape)
        combined = torch.cat((word_repr, per_sent_temp), dim=0)
        with_single_batch_dim = combined.flatten(0, 1)

        assert len(with_single_batch_dim.shape) == 3
        assert with_single_batch_dim.shape[0] == len(templates) * len(sentences)
        assert with_single_batch_dim.shape[1] == ELMO_SIZE + TEMPLATE_ENCODING_SIZE
        output, h = self.encoder(with_single_batch_dim)
        edges = bidirectional_rnn_output_edges(output)
        slot_match_log_probs = self.slot_match_predictor(output)
        template_match_log_probs = self.template_match_predictor(edges)
        # Index order is sentence, template, slot.
        slot_matches = slot_match_log_probs.view(len(sentences), len(templates),
                                                 TEMPLATE_SLOT_SIZE)
        template_matches = template_match_log_probs.view(len(sentences),
                                                         len(templates))
        return template_matches, slot_matches
