from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class UMLSBERT:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
        self.model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")

    def batch_encode(self, texts, batch_size=16):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings)
    
# umlsbert = UMLSBERT()