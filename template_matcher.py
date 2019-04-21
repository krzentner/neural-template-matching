import re

import torch
import torch.nn as nn

from allennlp.modules.elmo import Elmo, batch_to_ids

OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

ELMO_SIZE = 2048
HIDDEN_SIZE = 1024
TEMPLATE_ENCODING_SIZE = 512
TEMPLATE_ENCODING_SIZE_DIR = int(TEMPLATE_ENCODING_SIZE / 2)
TEMPLATE_SLOT_SIZE = 1

H = False

SLOT_RE = re.compile(r'\<(.*?)\>')

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
        self.input_encoder = nn.GRU(ELMO_SIZE, TEMPLATE_ENCODING_SIZE_DIR,
                                    num_layers=2, batch_first=True,
                                    bidirectional=True)
        self.attention_adjust = nn.Linear(1, 1)
        # self.attention_adjust.weight[0, 0] = 1.0
        # self.attention_adjust.bias[0] = 0.0
        self.match_prob_adjust = nn.Linear(1, 1)
        # self.match_prob_adjust.weight[0, 0] = 1.0
        # self.match_prob_adjust.bias[0] = 0.0

    if H:
        self = TemplateMatcher()
        templates = [['My', 'name', 'is', '<name>', '.'],
                     ['<greeting>', 'call', 'me', '<name>', ',', 'thanks', '.']]

    def encode_templates(self, templates):
        embedded = self.elmo(batch_to_ids(templates))
        word_repr = torch.cat(embedded['elmo_representations'],
                                         dim=2)
        slot_repr = torch.zeros(word_repr.shape[0], word_repr.shape[1],
                                TEMPLATE_SLOT_SIZE)
        mask = torch.ones(word_repr.shape[0], word_repr.shape[1], 1)
        slot_indices = []
        for i, t in enumerate(templates):
            slot_indices.append({})
            for j, w in enumerate(t):
                match = SLOT_RE.match(w)
                if match is not None:
                    slot_name = match.group(1)
                    slot_indices[-1][slot_name] = j
                    slot_repr[i, j] = 1.
                    mask[i, j] = 0.
        slot_indices
        masked_repr = mask * word_repr
        template_pre_encoding = torch.cat((masked_repr, slot_repr), dim=2)
        output, h = self.template_encoder(template_pre_encoding)
        slot_encodings = []
        slot_names = []
        for template_index, slots in enumerate(slot_indices):
            slot_encodings.append([])
            slot_names.append([])
            for slot_name, index in slots.items():
                slot_encoded = output[template_index, index]
                slot_encodings[-1].append(slot_encoded)
                slot_names[-1].append(slot_name)
        slot_encodings[0][0].shape
        torch.stack(slot_encodings[0]).shape
        stacked_slots = [torch.stack(slots) for slots in slot_encodings]
        padded_slots = torch.nn.utils.rnn.pad_sequence(stacked_slots,
                                                       batch_first=True)
        assert len(padded_slots.shape) == 3
        assert padded_slots.shape[0] == len(templates)
        assert padded_slots.shape[1] == max(len(s) for s in slot_encodings)
        assert padded_slots.shape[2] == TEMPLATE_ENCODING_SIZE
        num_slots = torch.tensor([float(len(slots)) for slots in slot_names])
        num_slots
        slot_names
        return (padded_slots, num_slots), slot_names

    if H:
        templates = (padded_slots, num_slots)
        sentences = [['Call', 'me', 'KR', '.'],
                     ['I', 'am', 'called', 'KR', '.'],
                     ['I\'m', 'KR', '.']]

    def forward(self, templates, sentences):
        padded_slots, num_slots = templates
        embedded = self.elmo(batch_to_ids(sentences))
        word_repr = torch.cat(embedded['elmo_representations'], dim=2)
        output, h = self.input_encoder(word_repr)
        output_seq_last = output.transpose(1, 2)
        output.shape
        padded_slots.unsqueeze(0).shape
        output_seq_last.unsqueeze(1).shape
        attention = torch.matmul(padded_slots.unsqueeze(0), output_seq_last.unsqueeze(1))
        attention.shape
        attention = self.attention_adjust(attention.unsqueeze(-1)).squeeze(-1)
        attention.shape
        padded_slots.shape
        assert len(attention.shape) == 4
        assert attention.shape[0] == len(sentences)
        assert attention.shape[1] == padded_slots.shape[0] # Number of templates
        assert attention.shape[2] == padded_slots.shape[1] # Number of slots
        assert attention.shape[3] == max(len(sent) for sent in sentences)

        match_prob = (attention.sum(dim=3).sum(dim=2) / num_slots).unsqueeze(-1)
        match_prob = self.match_prob_adjust(match_prob).squeeze(-1)
        return attention, match_prob
