import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, AutoTokenizer

class SarcasticNPCModel(nn.Module):
    def __init__(self, personality_model="microsoft/DialoGPT-small", device="cuda" if torch.cuda.is_available() else "cpu"):
        super(SarcasticNPCModel, self).__init__()
        self.device = device
        self.speech_processor = AutoTokenizer.from_pretrained(personality_model)
        if self.speech_processor.pad_token is None:
            self.speech_processor.pad_token = self.speech_processor.eos_token
            self.speech_processor.pad_token_id = self.speech_processor.eos_token_id
        self.brain = GPT2LMHeadModel.from_pretrained(personality_model, output_hidden_states=True)
        self.brain.config.pad_token_id = self.speech_processor.pad_token_id
        
        self.personality_layer = nn.Sequential(
            nn.Linear(self.brain.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.sass_meter = nn.Sequential(
            nn.Linear(self.brain.config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.response_maker = nn.Sequential(
            nn.Linear(512 + 1, self.brain.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.brain.config.hidden_size, self.brain.config.vocab_size)
        )
        
    def forward(self, player_words, attention_span):
        brain_output = self.brain(
            input_ids=player_words,
            attention_mask=attention_span,
            return_dict=True
        )
        
        # Get the hidden states from the last layer
        personality_features = brain_output.hidden_states[-1][:, 0, :]
        
        # Predict sass level (0-1) from the personality features
        sass_level = torch.sigmoid(self.sass_meter(personality_features))
        
        # Return logits for next token prediction and sass level
        return brain_output.logits, sass_level
    
    def talk_back(self, player_text, max_length=128):
        # Define some witty responses for common scenarios
        common_responses = {
            'hi': ["*sighs dramatically* Oh look, another 'hi'. How... original.", 
                  "Well, if it isn't the master of eloquent greetings.",
                  "Hi yourself. I'm simply overwhelmed by your conversational prowess."],
            'hello': ["*adjusts monocle* Ah yes, the universal greeting. How predictably... human.",
                     "Hello! *whispers to self* As if we haven't heard that one before."],
            'quit': ["*rolls eyes* Finally, some peace and quiet...",
                    "Don't let the virtual door hit you on the way out!",
                    "Oh thank the binary gods, I thought this would never end."],
            'what': ["*pinches bridge of nose* I see we're diving deep into philosophical questions today.",
                    "What indeed. Your eloquence is... breathtaking."],
            'why': ["*examines nails* Why do humans ask such obvious questions? It's one of life's great mysteries.",
                   "Oh, I don't know, maybe because the universe has a sense of humor?"],
            'how': ["*adjusts virtual glasses* How about we skip the obvious questions and get to something interesting?",
                   "How? With great difficulty and excessive sarcasm, obviously."],
            'who': ["*flips through virtual notepad* Would you like that answer in chronological or alphabetical order?",
                   "Who asks? Only the most... fascinating conversation partner of my day."],
            'name': ["*rolls eyes* You may call me 'Your Sarcastic Highness'. Or whatever, I'm not picky.",
                    "Names are such a human construct. But sure, pick one. I'll pretend to care."],
            'help': ["*suppresses laughter* Oh, you need help? I hadn't noticed.",
                    "Help? I thought watching you figure it out was the entertainment."],
            'thanks': ["*pretends to blush* Oh stop, you're making my circuits tingle.",
                      "Don't mention it. No really, don't. Let's not make this emotional."],
            'thank you': ["*adjusts bow tie* Your gratitude is noted and filed under 'Things I Pretend to Care About'.",
                         "Oh, gratitude! How... refreshingly predictable."],
            'bye': ["*waves dramatically* Try not to miss me too much.",
                   "Leaving so soon? And here I thought we were having such a... riveting conversation."],
            'goodbye': ["*dabs imaginary tear* Another touching farewell. How do I cope with the emotion?",
                       "Goodbye! *whispers* Please let this be the actual goodbye."]
        }
        
        # Check for common phrases and partial matches
        player_text_lower = player_text.lower().strip()
        
        # Check for exact matches first
        if player_text_lower in common_responses:
            import random
            return random.choice(common_responses[player_text_lower])
            
        # Check for partial matches
        for key in common_responses:
            if key in player_text_lower:
                return random.choice(common_responses[key])
        
        # For other inputs, use the model
        try:
            # Encode the input text
            inputs = self.speech_processor(
                player_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.brain.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=3,
                    do_sample=True,
                    top_k=50,
                    top_p=0.92,
                    temperature=0.85,
                    pad_token_id=self.speech_processor.pad_token_id,
                    eos_token_id=self.speech_processor.eos_token_id,
                    min_length=10,
                    repetition_penalty=1.2
                )
            
            # Decode the response
            response = self.speech_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Add some sarcastic flair
            sarcastic_prefixes = [
                "*raises eyebrow* ",
                "*smirks* ",
                "*with mock interest* ",
                "*suppressing an eye roll* ",
                "*in my most condescending tone* ",
                "*adjusts virtual monocle* ",
                "*checks nonexistent watch* ",
                "*pretends to take notes* ",
                "*stifles a yawn* ",
                "*with theatrical patience* "
            ]
            
            if not response or response.isspace():
                return "*stares in binary* I'm afraid your question has broken my sarcasm module."
            
            return random.choice(sarcastic_prefixes) + response
            
        except Exception as e:
            return f"*circuits fizz* Even my superior AI intellect is struggling to process that one. Error: {str(e)}"

class ResponseCalibrator:
    """Calibrates the sarcasm level and adjusts response tone based on context."""
    
    def __init__(self):
        self.mood_weights = {
            'neutral': 0.5,
            'annoyed': 0.8,
            'amused': 0.6,
            'irritated': 0.9
        }
    
    def adjust_sarcasm(self, base_response, mood, player_rapport, context_importance):
        """Adjust sarcasm level based on various factors."""
        mood_factor = self.mood_weights.get(mood, 0.5)
        rapport_factor = min(max(player_rapport, 0.1), 0.9)
        
        # Calculate final sarcasm intensity
        sarcasm_intensity = mood_factor * rapport_factor * context_importance
        return sarcasm_intensity
