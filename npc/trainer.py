import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from typing import Dict, Any

from .model import SarcasticNPCModel
from .data_handler import DialogueDataset, create_chat_loader

class NPCTrainer:
    def __init__(
        self,
        model: SarcasticNPCModel,
        train_dataset: DialogueDataset,
        val_dataset: DialogueDataset,
        config: Dict[str, Any]
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.train_loader = create_chat_loader(
            train_dataset,
            batch_size=config['chat_batch_size'],
            shuffle=True
        )
        self.val_loader = create_chat_loader(
            val_dataset,
            batch_size=config['chat_batch_size'],
            shuffle=False
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.sass_criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_speed'],
            weight_decay=config['brain_cleanup']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        
        self.writer = SummaryWriter(config['brain_logs'])
        self.config = config
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_sass_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch in progress_bar:
            player_text = batch['player_text'].to(self.device)
            player_attention = batch['player_attention'].to(self.device)
            npc_response = batch['npc_response'].to(self.device)
            sass_level = batch['sass_level'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            response_logits, predicted_sass = self.model(player_text, player_attention)
            
            # Reshape logits and targets for loss calculation
            response_logits = response_logits.view(-1, response_logits.size(-1))
            npc_response = npc_response.view(-1)
            
            # Calculate losses
            response_loss = self.criterion(response_logits, npc_response)
            sass_loss = self.sass_criterion(predicted_sass.squeeze(), sass_level)
            
            # Combined loss
            loss = response_loss + self.config['sass_importance'] * sass_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['thought_limit'])
            self.optimizer.step()
            
            total_loss += loss.item()
            total_sass_loss += sass_loss.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'sass_loss': sass_loss.item()
            })
        
        return {
            'loss': total_loss / len(self.train_loader),
            'sass_loss': total_sass_loss / len(self.train_loader)
        }
    
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_sass_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                player_text = batch['player_text'].to(self.device)
                player_attention = batch['player_attention'].to(self.device)
                npc_response = batch['npc_response'].to(self.device)
                sass_level = batch['sass_level'].to(self.device)
                
                # Forward pass
                response_logits, predicted_sass = self.model(player_text, player_attention)
                
                # Reshape logits and targets for loss calculation
                response_logits = response_logits.view(-1, response_logits.size(-1))
                npc_response = npc_response.view(-1)
                
                # Calculate losses
                response_loss = self.criterion(response_logits, npc_response)
                sass_loss = self.sass_criterion(predicted_sass.squeeze(), sass_level)
                
                # Combined loss
                loss = response_loss + self.config['sass_importance'] * sass_loss
                
                total_loss += loss.item()
                total_sass_loss += sass_loss.item()
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_sass_loss': total_sass_loss / len(self.val_loader)
        }
    
    def train(self, num_epochs: int):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            for name, value in {**train_metrics, **val_metrics}.items():
                self.writer.add_scalar(name, value, epoch)
            
            self.scheduler.step(val_metrics['val_loss'])
            
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, val_metrics['val_loss'])
            
            print(f"Epoch {epoch}: {train_metrics} {val_metrics}")
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        checkpoint_dir = self.config['memory_saves']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"brain_epoch_{epoch}_loss_{val_loss:.4f}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
