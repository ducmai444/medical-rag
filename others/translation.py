import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
from typing import Optional, Union, List
from langdetect import detect

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnViT5Translator:
    """
    A class for Vietnamese to English translation using EnViT5 model.
    
    This class provides an interface to translate text from Vietnamese to English
    using the VietAI/envit5-translation model fine-tuned with PEFT.
    
    Attributes:
        device (str): The device to run the model on ('cuda' or 'cpu')
        max_length (int): Maximum sequence length for translation
    """
    
    def __init__(self, peft_model_path: str = "ducmai-4203/envit5-medev-vi2en", max_length: int = 512):
        """
        Initialize the EnViT5 translator.
        
        Args:
            peft_model_path (str): Path to the PEFT model
            max_length (int): Maximum sequence length for translation
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        
        try:
            # Load base model
            logger.info("Loading base model...")
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/envit5-translation")
            self.base_model = self.base_model.to(self.device).half()
            
            # Load PEFT model
            logger.info("Loading PEFT model...")
            self.model = PeftModel.from_pretrained(
                self.base_model,
                peft_model_path
            ).to(self.device).half()
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained("VietAI/envit5-translation")
            
            logger.info("Model initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize the model: {str(e)}")
            raise
    
    def translate(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Translate text from Vietnamese to English, with automatic language detection.
        If text is already in English, it will be returned as is.
        Mixed Vietnamese-English text will be translated.
        
        Args:
            text (Union[str, List[str]]): Input text or list of texts
            
        Returns:
            Union[str, List[str]]: Translated or original text(s)
        """
        try:
            # Convert single string to list for uniform processing
            is_single_string = isinstance(text, str)
            texts = [text] if is_single_string else text
            
            # Process each text
            texts_to_translate = []
            indices_to_translate = []
            result = [""] * len(texts)
            
            # Check each text for language and prepare for translation if needed
            for i, t in enumerate(texts):
                try:
                    # Try to detect language
                    lang = detect(t)
                    # If pure English, keep original
                    if lang == 'en':
                        result[i] = t
                    else:
                        # Vietnamese or mixed content needs translation
                        texts_to_translate.append(f"vi: {t}")
                        indices_to_translate.append(i)
                except:
                    # If detection fails, assume it needs translation
                    texts_to_translate.append(f"vi: {t}")
                    indices_to_translate.append(i)
            
            # If we have texts to translate
            if texts_to_translate:
                # Tokenize
                inputs = self.tokenizer(
                    texts_to_translate,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                # Generate translation
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_beams=2,
                        early_stopping=True
                    )
                
                # Decode outputs and put them in the right positions
                translations = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
                
                for trans_idx, orig_idx in enumerate(indices_to_translate):
                    result[orig_idx] = translations[trans_idx]
            
            # Return single string if input was single string
            return result[0] if is_single_string else result
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return "" if is_single_string else [""] * len(texts)
    
    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Make the class callable for easy translation.
        
        Args:
            text (Union[str, List[str]]): Input text or list of texts
            
        Returns:
            Union[str, List[str]]: Translated or original text(s)
        """
        return self.translate(text)

# Example usage:
# translator = EnViT5Translator()
# vietnamese_text = translator.translate("Sốt là dấu hiệu thường gặp khi cơ thể bị nhiễm trùng")
# english_text = translator.translate("The patient has a fever")  # Will return as is
# mixed_text = translator.translate("Patient has sốt cao và đau đầu")  # Will translate
