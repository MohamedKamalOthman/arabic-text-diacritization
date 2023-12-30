import torch


def batch_accuracy(
    output: torch.Tensor, gold: torch.Tensor, pad_index: int = -1, device="cuda"
):
    """
    Calculates the accuracy of the model

    :param output: the output of the model
    :param gold: the gold diacritic sequence
    :param pad_index: the index of the padding token
    :param device: the device to use
    :return: the accuracy of the model
    """
    predictions = output.argmax(dim=1, keepdim=True)
    non_pad_elements = torch.nonzero((gold != pad_index))
    correct = predictions[non_pad_elements].squeeze(1).eq(gold[non_pad_elements])
    return correct.sum() / torch.FloatTensor([gold[non_pad_elements].shape[0]]).to(
        device
    )


def batch_diac_error(
    char_seq, output: torch.Tensor, gold: torch.Tensor, arabic_ids, device="cuda"
):
    """
    Calculates the number of true and false predictions for diacritics

    :param char_seq: the input character sequence
    :param output: the output of the model
    :param gold: the gold diacritic sequence
    :param arabic_ids: the ids of the arabic letters
    :param device: the device to use
    :return: the number of true predictions, the number of false predictions
    """
    char_seq = char_seq.contiguous().view(-1).flatten()
    # only calculate DER for arabic letters meaning ignore any other characters while getting true and false counts
    # from char_seq use the arabic ids to mask out non arabic letters to ignore them
    arabic_ids = torch.tensor(arabic_ids, device=device)
    arabic_mask = torch.isin(char_seq, arabic_ids)

    predictions = output.argmax(dim=1, keepdim=True)
    arabic_elements = torch.nonzero(arabic_mask)
    correct = (
        predictions[arabic_elements].flatten() == gold[arabic_elements].flatten()
    ).flatten()

    # return number of true predictions, number of false predictions
    return correct.sum(), (correct.shape[0] - correct.sum())
