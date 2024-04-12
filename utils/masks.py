import numpy as np
import torch

import operator


def to_mask(n_data, indices):
    mask = torch.zeros(n_data, dtype=bool)
    mask[indices] = 1

    return mask


def merge_mask(mask_1, mask_2):
    mask_3 = mask_1 | mask_2
    return mask_3


def multiply_round(n_data, cfg):
    s_total = sum(cfg.values())
    sizes = {name: int(s * n_data / s_total) for name, s in cfg.items()}

    max_name = max(sizes.items(), key=operator.itemgetter(1))[0]
    sizes[max_name] += n_data - sum(sizes.values())

    return sizes


def generate_masks(n_data, split_config):
    if "shadow1" in split_config:
        assert type(split_config) is dict
        assert "public" in split_config and "private" in split_config
        assert type(split_config["private"]) is dict

        permutation = np.random.permutation(n_data)
        if type(split_config["public"]) is dict:
            n_public = round(sum(split_config["public"].values()) * n_data)
            print(n_public)
        else:
            n_public = int(split_config["public"] * n_data)
        n_private = int(sum(split_config["private"].values()) * n_data)
        n_private = n_public + n_private
        print(n_private)
        n_shadow_1 = int(sum(split_config["shadow1"].values()) * n_data)
        n_shadow_1 = n_private + n_shadow_1
        print(n_shadow_1)

        # allocate all the data in dataset
        if n_shadow_1 != n_data:
            n_public = n_public+1
            n_private = n_private+1
            n_shadow_1 = n_shadow_1+1

        # mask the data
        known_masks = {}
        known_masks["public"] = to_mask(n_data, permutation[:n_public])
        known_masks["private"] = to_mask(n_data, permutation[n_public:n_private])
        known_masks["shadow1"] = to_mask(n_data, permutation[n_private:n_shadow_1])
        hidden_masks = {}
        hidden_masks["private"] = {}
        sizes = multiply_round(n_private-n_public, split_config["private"])
        print(' Private', sizes)
        offset = n_public
        for name, size in sizes.items():
            hidden_masks["private"][name] = to_mask(n_data, permutation[offset:offset + size])
            offset += size

        hidden_masks["shadow1"] = {}
        sizes = multiply_round(n_shadow_1-n_private, split_config["shadow1"])
        print(' shadow1', sizes)
        for name, size in sizes.items():
            hidden_masks["shadow1"][name] = to_mask(n_data, permutation[offset:offset + size])
            offset += size

        print(offset)
        assert offset == n_data

        if type(split_config["public"]) is dict:
            hidden_masks["public"] = {}
            public_sizes = multiply_round(n_public, split_config["public"])
            print('Public', public_sizes)
            public_offset = 0
            for name, size in public_sizes.items():
                hidden_masks["public"][name] = to_mask(n_data, permutation[public_offset:public_offset + size])
                public_offset += size
            assert public_offset == n_public
        else:
            hidden_masks["public"] = known_masks["public"]
    else:
        assert type(split_config) is dict
        assert "public" in split_config and "private" in split_config
        assert type(split_config["private"]) is dict

        permutation = np.random.permutation(n_data)
        if type(split_config["public"]) is dict:
            n_public=int(sum(split_config["public"].values())*n_data)
        else:
            n_public = int(split_config["public"] * n_data)
        n_private = n_data - n_public
        known_masks = {}
        known_masks["public"] = to_mask(n_data, permutation[:n_public])
        known_masks["private"] = to_mask(n_data, permutation[n_public:])
        hidden_masks = {}
        hidden_masks["private"] = {}

        sizes = multiply_round(n_private, split_config["private"])
        print(' Private', sizes)
        offset = n_public
        for name, size in sizes.items():
            hidden_masks["private"][name] = to_mask(n_data, permutation[offset:offset+size])
            offset += size

        assert offset == n_data

        if type(split_config["public"]) is dict:
            hidden_masks["public"] = {}
            public_sizes = multiply_round(n_public, split_config["public"])
            print('Public', public_sizes)
            public_offset = 0
            for name, size in public_sizes.items():
                hidden_masks["public"][name] = to_mask(n_data, permutation[public_offset:public_offset+size])
                public_offset += size
            assert public_offset == n_public
        else:
            hidden_masks["public"] = known_masks["public"]

    return known_masks, hidden_masks


def regenerate_mask(mask1, mask2, mask3):
    # Combine the masks to find all true indices
    combined_mask = mask1 | mask2 | mask3
    true_indices = combined_mask.nonzero(as_tuple=False).view(-1)
    print(true_indices)

    # Shuffle the true indices
    shuffled_indices = true_indices[torch.randperm(len(true_indices))]

    # Initialize the new masks with zeros
    new_masks = [torch.zeros_like(mask) for mask in [mask1, mask2, mask3]]

    # Allocate shuffled indices to each new mask without overlap
    start = 0
    for i, mask in enumerate([mask1, mask2, mask3]):
        true_count = int(torch.sum(mask))
        end = start + true_count
        new_masks[i][shuffled_indices[start:end]] = 1
        start = end

    return tuple(new_masks)


def evaluate_masks(train_score, heldout_score, threshold, attack_base=None):

    if attack_base=='loss' or attack_base=='mean':
        true_positives = (train_score <= threshold).float()
        false_negatives= (train_score > threshold).float()
        true_negatives = (heldout_score > threshold).float()
        false_positives = (heldout_score <= threshold).float()
    else:
        true_positives = (train_score >= threshold).float()
        false_negatives = (train_score < threshold).float()
        true_negatives = (heldout_score < threshold).float()
        false_positives = (heldout_score >= threshold).float()

    tpr=torch.sum(true_positives) / (torch.sum(true_positives) + torch.sum(false_negatives))
    fpr=torch.sum(false_positives) / (torch.sum(false_positives) + torch.sum(true_negatives))
    recall = torch.sum(true_positives)
    precision = torch.sum(true_positives) / (torch.sum(true_positives) + torch.sum(false_positives))

    accuracy = (torch.sum(true_positives) + torch.sum(true_negatives)) / (torch.sum(true_positives) + torch.sum(false_positives) + torch.sum(true_negatives) + torch.sum(false_negatives))

    return tpr, fpr, precision, recall, accuracy
