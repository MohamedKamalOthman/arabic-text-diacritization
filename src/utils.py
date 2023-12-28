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
