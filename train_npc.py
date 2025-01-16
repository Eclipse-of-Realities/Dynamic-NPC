import os
import json
from npc.model import SarcasticNPCModel
from npc.data_handler import DialogueDataset, create_chat_loader
from npc.trainer import NPCTrainer
from transformers import AutoTokenizer
import shutil

def main():
    config = {
        'personality_model': 'microsoft/DialoGPT-small',
        'chat_batch_size': 8,
        'learning_speed': 2e-5,
        'brain_cleanup': 0.01,
        'thought_limit': 1.0,
        'sass_importance': 0.3,
        'num_epochs': 10,
        'brain_logs': 'logs',
        'memory_saves': 'checkpoints',
        'chat_examples': 'data/sample_dialogues.json'
    }
    
    os.makedirs(config['brain_logs'], exist_ok=True)
    os.makedirs(config['memory_saves'], exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(config['personality_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    npc_model = SarcasticNPCModel(config['personality_model'])
    
    with open(config['chat_examples'], 'r') as file:
        all_dialogues = json.load(file)
    
    dialogues = all_dialogues['dialogues']
    training_size = int(0.8 * len(dialogues))
    
    training_dialogues = {'dialogues': dialogues[:training_size]}
    testing_dialogues = {'dialogues': dialogues[training_size:]}
    
    os.makedirs('data/temp', exist_ok=True)
    
    with open('data/temp/training.json', 'w') as file:
        json.dump(training_dialogues, file)
    with open('data/temp/testing.json', 'w') as file:
        json.dump(testing_dialogues, file)
    
    training_dataset = DialogueDataset('data/temp/training.json', tokenizer)
    testing_dataset = DialogueDataset('data/temp/testing.json', tokenizer)
    
    trainer = NPCTrainer(npc_model, training_dataset, testing_dataset, config)
    
    trainer.train(config['num_epochs'])
    
    # Clean up temporary files
    shutil.rmtree('data/temp', ignore_errors=True)

if __name__ == '__main__':
    main()
