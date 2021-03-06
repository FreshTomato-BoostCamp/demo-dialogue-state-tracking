
import torch


class CFG:
    Train: str = "./input/data/train_dataset"
    Sample: str = "./input/data/test.json"
    TrainSlotMeta: str = "./input/data/train_dataset/slot_meta.json"
    TrainOntology: str = "./input/data/train_dataset/ontology.json"
    TrainDials: str = "../input/data/train_dataset/train_dials.json"

    Test: str= "./input/data/eval_dataset"
    EvalSlotMeta: str = "./input/data/eval_dataset/slot_meta.json"
    EvalDials: str = "./input/data/eval_dataset/train_dials.json"

    Models: str = "./models"
    Output: str = "./output"
    Device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PreTrainedBert: str = 'kykim/bert-kor-base'