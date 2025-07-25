from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class UMLSBERT:
    # def __init__(self):
    #     self.tokenizer = AutoTokenizer.from_pretrained("GanjinZero/UMLSBert_ENG")
    #     self.model = AutoModel.from_pretrained("GanjinZero/UMLSBert_ENG")

    # def mean_pooling(self, model_output, attention_mask):
    #     token_embeddings = model_output[0]  
    #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # def batch_encode(self, texts, batch_size=16):
    #     all_embeddings = []
    #     for i in range(0, len(texts), batch_size):
    #         batch_texts = texts[i:i+batch_size]
    #         inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    #         with torch.no_grad():
    #             model_output = self.model(**inputs)
    #             attention_mask = inputs["attention_mask"]
    #             batch_embeddings = self.mean_pooling(model_output, attention_mask)
    #         all_embeddings.extend(batch_embeddings)  
    #     return np.array(all_embeddings)
    
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