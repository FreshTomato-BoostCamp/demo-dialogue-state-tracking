from typing import *
from copy import deepcopy
from data_utils import OpenVocabDSTFeature, DSTInputExample
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

# NOTE. 시스템 우선 대화 기반으로 구현함
class DialogueDirector:
    def __init__(self, model, preprocessor, device: str=device):
        self.__model = model
        self.__dialogue = self.__initialize_dialogue()
        self.__preprocessor = preprocessor
        self.__device = device
        self.__states = None
        
    def record(self, dialogue_pair: list) -> DSTInputExample:
        # extract incoming information
        dial_dict = dict()
        for p in dialogue_pair:
            if p['role'] == 'user':
                dial_dict['user'] = p
            else:
                dial_dict['sys'] = p

        # update current_turn & context_turns
        current_turn = self.__dialogue.current_turn
        incoming_turn = [dial_dict['sys']['text'], dial_dict['user']['text']]
        self.__dialogue.context_turns.extend(deepcopy(current_turn))
        self.__dialogue.current_turn = deepcopy(incoming_turn)

        # update guid, dialogue order ID
        self.__dialogue.guid += 1
        self.__need_update = True

        # get dialogue state
        self.__states = self.__get_state()

    @property
    def states(self):
        return self.__states

    def reset_dialogue(self): # 대화 종료
        """Reset dialogue instance. Used when dialogue has been done"""
        self.__dialogue = self.__initialize_dialogue()
        self.__states = None

    def __get_state(self) -> list:
        x = self.__preprocessor._convert_example_to_feature(self.__dialogue)
        x = self.__preprocessor.collate_fn([x])
        input_id, segment_id, input_mask, _, _, _ = [
            b.to(self.__device) if not isinstance(b, list) else b for b in x
        ]
        with torch.no_grad():
            o, g = self.__model(input_id, segment_id, input_mask, 9)
            _, generated_id = o.max(-1)
            _, gated_id = g.max(-1)
        
        states = self.__preprocessor.recover_state(
            gated_id.squeeze(0).tolist(),
            generated_id.squeeze(0).tolist()
            )
        states = postprocess_state(states)
        return states

    def __initialize_dialogue(self):
        dialogue = DSTInputExample(
        guid=0,
        current_turn=[],
        context_turns=[],
        label=[]
        )
        return dialogue


def postprocess_state(state) -> list:
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
    return state