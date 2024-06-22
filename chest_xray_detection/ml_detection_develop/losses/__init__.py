from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss

losses_dict = {
    "BCELoss": BCELoss,
    "BCEWithLogitsLoss": BCEWithLogitsLoss,
    "CrossEntropyLoss": CrossEntropyLoss,
}
