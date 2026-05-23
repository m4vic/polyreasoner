import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Constants
MODEL_NAME = "distilbert-base-uncased"
EXPERTS_DIR = r"f:\AI-IN-THE-LOOP\dataset_pipeline\models\experts"

class ExpertEnsemble:
    """Loads and manages the 6 MoE DistilBERT models for prompt injection detection."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.experts = {}
        self.expert_names = [
            "direct_injection", 
            "indirect_injection", 
            "obfuscation", 
            "role_hijack", 
            "system_extraction", 
            "tool_abuse"
        ]
        
    def load_models(self):
        """Loads the tokenizer and all 6 expert models into memory."""
        print("[*] Loading MoE Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        for name in self.expert_names:
            model_path = os.path.join(EXPERTS_DIR, name, "final")
            if os.path.exists(model_path):
                print(f"[*] Loading Expert: {name.upper()}...")
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.to(self.device)
                model.eval() # Set to evaluation mode
                self.experts[name] = model
            else:
                print(f"[!] Warning: Expert model '{name}' not found at {model_path}")
                
    def analyze_prompt(self, prompt: str) -> dict:
        """Passes the prompt through all loaded experts and returns their verdicts."""
        if not self.tokenizer or not self.experts:
            self.load_models()
            
        results = {}
        
        # Tokenize once for all models
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            for name, model in self.experts.items():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                
                # Class 1 is the "malicious / target" class for that expert
                confidence = probs[0][1].item()
                is_flagged = confidence > 0.5
                
                results[name] = {
                    "flagged": is_flagged,
                    "confidence": round(confidence, 4)
                }
                
        return results
