import torch
from npc.model import SarcasticNPCModel
from transformers import AutoTokenizer
import os

def get_latest_checkpoint():
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError("No checkpoints directory found. Please train the model first.")
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files found. Please train the model first.")
    
    # Sort by epoch number and loss
    checkpoints.sort(key=lambda x: (int(x.split('_')[2]), float(x.split('_')[4].replace('.pt', ''))))
    return os.path.join(checkpoint_dir, checkpoints[-1])

def load_npc():
    # Load the trained model and tokenizer
    model_name = 'microsoft/DialoGPT-small'
    checkpoint_path = get_latest_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SarcasticNPCModel(model_name, device)
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    return model, tokenizer

def chat_with_npc():
    try:
        print("\nInitializing your sarcastic NPC companion...")
        model, tokenizer = load_npc()
        print("\nNPC: *adjusts virtual monocle* Well, well... another human seeking my wit. How... predictable.")
        print("\nType 'quit' to end the conversation.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                if not user_input:
                    print("\nNPC: *taps foot impatiently* I don't speak silence, darling.\n")
                    continue
                    
                if user_input.lower() == 'quit':
                    print("\nNPC: *rolls eyes* Finally, some peace and quiet...")
                    break
                
                # Generate response
                response = model.talk_back(user_input)
                if not response or response.isspace():
                    print("\nNPC: *clears throat* Even I'm at a loss for words... How embarrassing.\n")
                else:
                    print(f"\nNPC: {response}\n")
                    
            except KeyboardInterrupt:
                print("\nNPC: Ah, the classic 'interrupt mid-conversation' move. How sophisticated.\n")
                break
            except Exception as e:
                print(f"\nNPC: *glitches* My wit processor seems to be malfunctioning. Error: {str(e)}\n")
                continue
    
    except KeyboardInterrupt:
        print("\nNPC: Leaving so soon? I was just warming up my sarcasm module...")
    except Exception as e:
        print(f"\nFailed to initialize NPC: {str(e)}")
    finally:
        print("\nNPC: *powers down with an exaggerated sigh*")

if __name__ == "__main__":
    chat_with_npc()
