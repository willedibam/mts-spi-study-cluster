"""Schedule-only diagnostics; never inputs to the waveform/representation models."""
import numpy as np


DISCORDANT_BLOCKS = (1, 10, 13, 22)


def fit_block_lookup(blocks, y):
    """Source-category labels only; missing positions use an untuned 0.5 prior."""
    blocks, y = np.asarray(blocks), np.asarray(y)
    if blocks.shape != y.shape or y.ndim != 1 or not len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected aligned source block IDs and binary labels')
    return {int(block): float(np.mean(y[blocks == block])) for block in np.unique(blocks)}


def predict_block_lookup(lookup, blocks):
    return np.asarray([lookup.get(int(block), .5) for block in blocks])


def discordant_mask(blocks, y, participants):
    """Use all swapped-position observations for people retaining both classes.

    Membership depends on the frozen schedule and QC/label availability, never
    model scores. Apply one identical mask to every method and report exclusions.
    """
    blocks, y, participants = map(np.asarray, (blocks, y, participants))
    if blocks.ndim != 1 or blocks.shape != y.shape or y.shape != participants.shape:
        raise ValueError('Expected aligned block, label and participant vectors')
    selected = np.isin(blocks, DISCORDANT_BLOCKS)
    eligible = [person for person in np.unique(participants)
                if set(np.unique(y[selected & (participants == person)])) == {0, 1}]
    return selected & np.isin(participants, eligible)
