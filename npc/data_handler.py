import json
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class DialogueDataset(Dataset):
    def __init__(self, dialogue_file: str, tokenizer, max_words: int = 128):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.dialogues = self.read_conversations(dialogue_file)
        
    def read_conversations(self, dialogue_file: str) -> List[Dict]:
        with open(dialogue_file, 'r', encoding='utf-8') as file:
            raw_conversations = json.load(file)
            
        clean_conversations = []
        for chat in raw_conversations['dialogues']:
            clean_chat = {
                'player_text': chat['player_input'],
                'scene': chat.get('context', ''),
                'npc_mood': chat.get('mood', 'neutral'),
                'sass_level': chat.get('sarcasm_level', 0.5),
                'npc_response': chat['npc_response']
            }
            clean_conversations.append(clean_chat)
            
        return clean_conversations
    
    def __len__(self) -> int:
        return len(self.dialogues)
    
    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        
        player_text = dialogue['player_text']
        npc_response = dialogue['npc_response']
        sass_level = torch.tensor(dialogue['sass_level'], dtype=torch.float32)
        
        player_words = self.tokenizer(
            player_text,
            padding='max_length',
            max_length=self.max_words,
            truncation=True,
            return_tensors='pt'
        )
        
        npc_words = self.tokenizer(
            npc_response,
            padding='max_length',
            max_length=self.max_words,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'player_text': player_words['input_ids'].squeeze(0),
            'player_attention': player_words['attention_mask'].squeeze(0),
            'npc_response': npc_words['input_ids'].squeeze(0),
            'sass_level': sass_level
        }

class SceneAnalyzer:
    def __init__(self):
        self.scene_types = {
            'quest_state': ['not_started', 'ongoing', 'failed', 'completed'],
            'player_attitude': ['friendly', 'hostile', 'neutral'],
            'chat_history': ['first_time', 'repeated', 'frequent']
        }
    
    def analyze_scene(self, game_state: Dict) -> np.ndarray:
        scene_info = []
        
        quest_state = game_state.get('quest_state', 'not_started')
        scene_info.extend(self._convert_to_numbers(quest_state, self.scene_types['quest_state']))
        
        attitude = game_state.get('player_attitude', 'neutral')
        scene_info.extend(self._convert_to_numbers(attitude, self.scene_types['player_attitude']))
        
        history = game_state.get('chat_history', 'first_time')
        scene_info.extend(self._convert_to_numbers(history, self.scene_types['chat_history']))
        
        return np.array(scene_info)
    
    def _convert_to_numbers(self, value: str, categories: List[str]) -> List[float]:
        return [1.0 if cat == value else 0.0 for cat in categories]

def create_chat_loader(
    dataset: DialogueDataset,
    batch_size: int = 16,
    shuffle: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )