import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Constants
MODEL_NAME = "distilbert-base-uncased"
EXPERTS_DIR = os.getenv("POLYREASONER_EXPERTS_DIR", r"f:\AI-IN-THE-LOOP\dataset_pipeline\models\experts")

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
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        except Exception as e:
            print(f"[!] Failed to load tokenizer locally, falling back to HF Hub: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Reload env variable or settings in case it was set after init
        experts_dir = os.getenv("POLYREASONER_EXPERTS_DIR", EXPERTS_DIR)

        for name in self.expert_names:
            model_path = os.path.join(experts_dir, name, "final")
            if os.path.exists(model_path):
                print(f"[*] Loading local Expert: {name.upper()} from {model_path}...")
                model_to_load = model_path
            else:
                # E.g. neuralchemy/distilbert-expert-direct-injection-threat-matrix
                hf_name = name.replace("_", "-")
                hf_repo = f"neuralchemy/distilbert-expert-{hf_name}-threat-matrix"
                print(f"[!] Expert model '{name}' not found locally. Loading from Hugging Face: '{hf_repo}'...")
                model_to_load = hf_repo

            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_to_load)
                model.to(self.device)
                model.eval() # Set to evaluation mode
                self.experts[name] = model
            except Exception as e:
                print(f"[!] ERROR: Failed to load expert model '{name}' from {model_to_load}: {e}")
                
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
