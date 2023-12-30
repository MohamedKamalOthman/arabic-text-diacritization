import torch


def batch_accuracy(
    output: torch.Tensor, gold: torch.Tensor, pad_index: int = -1, device="cuda"
):
    predictions = output.argmax(dim=1, keepdim=True)
    non_pad_elements = torch.nonzero((gold != pad_index))
    correct = predictions[non_pad_elements].squeeze(1).eq(gold[non_pad_elements])
    return correct.sum() / torch.FloatTensor([gold[non_pad_elements].shape[0]]).to(
        device
    )


def calc_accuracy(
    output: torch.Tensor, gold: torch.Tensor, pad_index: int = -1, device="cuda"
):
    predictions = output.argmax(dim=1, keepdim=True)
    non_pad_elements = torch.nonzero((gold != pad_index))
    correct = predictions[non_pad_elements].squeeze(1).eq(gold[non_pad_elements])
    # return number of true predictions, number of false predictions
    return correct.sum(), (gold[non_pad_elements].shape[0] - correct.sum())


def calc_der(
    char_seq, output: torch.Tensor, gold: torch.Tensor, arabic_ids, device="cuda"
):
    # only calculate DER for arabic letters meaning ignore any other characters while getting true and false counts
    # from char_seq use the arabic ids to mask out non arabic letters to ignore them
    arabic_ids = torch.tensor(arabic_ids, device=device)
    arabic_mask = torch.isin(char_seq, arabic_ids)

    predictions = output.argmax(dim=2, keepdim=True).squeeze(2)
    arabic_elements = torch.nonzero(arabic_mask)
    # print shape of predictions and gold
    # print(predictions.shape, gold.shape)
    
    correct = predictions[arabic_mask] == gold[arabic_mask]

    # return number of true predictions, number of false predictions
    return correct.sum(), (correct.shape[0] - correct.sum())
