"""
Medical Knowledge Pipeline for RAG system.
Integrates Vietnamese translation, medical NER, UMLS API, and MMR ranking.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
import time

# Import medical modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'others'))

from others.translation import EnViT5Translator
from others.ner import MedicalNERLLM
from others.umls import UMLS_API
from others.ranking import MMR_reranking, similarity_score
from others.umlsbert import UMLSBERT

logger = logging.getLogger(__name__)

@dataclass
class MedicalEntity:
    """Represents a medical entity extracted from text."""
    term: str
    cui: str = ""
    name: str = ""
    confidence: float = 0.0
    entity_type: str = ""

@dataclass
class UMLSResult:
    """Represents a UMLS knowledge result."""
    cui: str
    name: str
    relation_label: str
    related_concept: str
    score: float = 0.0
    source: str = "UMLS"

@dataclass
class MedicalContext:
    """Combined medical context for RAG."""
    original_query: str
    translated_query: str
    medical_entities: List[MedicalEntity]
    umls_results: List[UMLSResult]
    confidence_score: float
    processing_time: float

class MedicalRAGPipeline:
    """
    Comprehensive medical pipeline for Vietnamese medical Q&A.
    """
    
    def __init__(self, 
                 umls_api_key: str,
                 enable_translation: bool = True,
                 enable_medical_ner: bool = True,
                 enable_umls: bool = True,
                 max_umls_results: int = 3):
        """
        Initialize medical pipeline components.
        
        Args:
            umls_api_key: API key for UMLS access
            enable_translation: Enable Vietnamese-English translation
            enable_medical_ner: Enable medical entity extraction
            enable_umls: Enable UMLS knowledge retrieval
            max_umls_results: Maximum UMLS results to return
        """
        self.umls_api_key = umls_api_key
        self.enable_translation = enable_translation
        self.enable_medical_ner = enable_medical_ner
        self.enable_umls = enable_umls
        self.max_umls_results = max_umls_results
        
        # Initialize components
        self._init_components()
        
    def _init_components(self):
        """Initialize all medical pipeline components."""
        try:
            # Translation component
            if self.enable_translation:
                logger.info("Initializing Vietnamese-English translator...")
                try:
                    self.translator = EnViT5Translator()
                    logger.info("✅ Translator initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize translator: {e}")
                    self.translator = None
                    self.enable_translation = False
            
            # Medical NER component
            if self.enable_medical_ner:
                logger.info("Initializing Medical NER...")
                try:
                    self.medical_ner = MedicalNERLLM()
                    logger.info("✅ Medical NER initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Medical NER: {e}")
                    self.medical_ner = None
                    self.enable_medical_ner = False
            
            # UMLS API component
            if self.enable_umls:
                logger.info("Initializing UMLS API...")
                try:
                    self.umls_api = UMLS_API(self.umls_api_key)
                    logger.info("✅ UMLS API initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize UMLS API: {e}")
                    self.umls_api = None
                    self.enable_umls = False
            
            # UMLS BERT for ranking
            if self.enable_umls:
                logger.info("Initializing UMLS BERT...")
                try:
                    self.umlsbert = UMLSBERT()
                    logger.info("✅ UMLS BERT initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize UMLS BERT: {e}")
                    self.umlsbert = None
            
            logger.info("Medical pipeline initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize medical pipeline: {e}")
            raise
    
    def process_query(self, query: str) -> MedicalContext:
        """
        Process a medical query through the complete pipeline.
        
        Args:
            query: Input query (Vietnamese or English)
            
        Returns:
            MedicalContext with all processed information
        """
        start_time = time.time()
        
        try:
            # Step 1: Language detection and translation
            translated_query = self._translate_query(query)
            
            # Step 2: Medical entity extraction
            medical_entities = self._extract_medical_entities(translated_query)
            
            # Step 3: UMLS knowledge retrieval
            umls_results = self._retrieve_umls_knowledge(translated_query, medical_entities)
            
            # Step 4: Calculate confidence score
            confidence_score = self._calculate_confidence(medical_entities, umls_results)
            
            processing_time = time.time() - start_time
            
            return MedicalContext(
                original_query=query,
                translated_query=translated_query,
                medical_entities=medical_entities,
                umls_results=umls_results,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to process medical query: {e}")
            # Return minimal context on error
            return MedicalContext(
                original_query=query,
                translated_query=query,
                medical_entities=[],
                umls_results=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time
            )
    
    def _translate_query(self, query: str) -> str:
        """Translate Vietnamese query to English if needed."""
        if not self.enable_translation or not self.translator:
            return query
        
        try:
            # Auto-detect and translate if needed
            translated = self.translator.translate(query)
            logger.debug(f"Translation: '{query}' -> '{translated}'")
            return translated
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return query
    
    def _extract_medical_entities(self, query: str) -> List[MedicalEntity]:
        """Extract medical entities using NER."""
        if not self.enable_medical_ner or not self.medical_ner:
            return []
        
        try:
            # Extract medical terms
            medical_terms = self.medical_ner.predict(query, min_score=0.5)
            
            entities = []
            for term in medical_terms:
                entity = MedicalEntity(
                    term=term,
                    confidence=0.8,  # Default confidence from NER
                    entity_type="medical_term"
                )
                entities.append(entity)
            
            logger.debug(f"Extracted {len(entities)} medical entities: {[e.term for e in entities]}")
            return entities
            
        except Exception as e:
            logger.warning(f"Medical NER failed: {e}")
            return []
    
    def _retrieve_umls_knowledge(self, query: str, entities: List[MedicalEntity]) -> List[UMLSResult]:
        """Retrieve and rank UMLS knowledge."""
        if not self.enable_umls or not self.umls_api:
            return []
        
        try:
            all_relations = []
            processed_cuis = set()
            
            # Get UMLS data for each medical entity
            for entity in entities:
                try:
                    # Search for CUI
                    cui_results = self.umls_api.search_cui(entity.term)
                    
                    if cui_results:
                        cui, name = cui_results[0]  # Take the first (best) result
                        entity.cui = cui
                        entity.name = name
                        
                        # Avoid duplicate CUI processing
                        if cui not in processed_cuis:
                            processed_cuis.add(cui)
                            
                            # Get relations for this CUI
                            relations = self.umls_api.get_relations(cui, pages=5)
                            if relations:
                                all_relations.extend(relations)
                                logger.debug(f"Retrieved {len(relations)} relations for {entity.term} (CUI: {cui})")
                
                except Exception as e:
                    logger.warning(f"Failed to process entity {entity.term}: {e}")
                    continue
            
            # Rank relations using MMR
            if all_relations and self.umlsbert:
                try:
                    ranked_relations = MMR_reranking(
                        query=query,
                        relations=all_relations,
                        top_k=self.max_umls_results
                    )
                    
                    # Convert to UMLSResult objects
                    umls_results = []
                    for i, rel in enumerate(ranked_relations):
                        result = UMLSResult(
                            cui=rel.get("relatedFromIdName", ""),
                            name=rel.get("relatedFromIdName", ""),
                            relation_label=rel.get("additionalRelationLabel", ""),
                            related_concept=rel.get("relatedIdName", ""),
                            score=1.0 - (i * 0.1),  # Decreasing score based on rank
                            source="UMLS"
                        )
                        umls_results.append(result)
                    
                    logger.info(f"Retrieved and ranked {len(umls_results)} UMLS results")
                    return umls_results
                    
                except Exception as e:
                    logger.warning(f"MMR ranking failed: {e}")
                    # Fallback to similarity-based ranking
                    return self._fallback_ranking(query, all_relations)
            
            return []
            
        except Exception as e:
            logger.error(f"UMLS knowledge retrieval failed: {e}")
            return []
    
    def _fallback_ranking(self, query: str, relations: List[Dict]) -> List[UMLSResult]:
        """Fallback ranking using similarity scores."""
        try:
            ranked_relations = similarity_score(
                query=query,
                relations=relations,
                top_k=self.max_umls_results
            )
            
            umls_results = []
            for i, rel in enumerate(ranked_relations):
                result = UMLSResult(
                    cui=rel.get("relatedFromIdName", ""),
                    name=rel.get("relatedFromIdName", ""),
                    relation_label=rel.get("additionalRelationLabel", ""),
                    related_concept=rel.get("relatedIdName", ""),
                    score=0.8 - (i * 0.1),  # Slightly lower scores for fallback
                    source="UMLS_fallback"
                )
                umls_results.append(result)
            
            return umls_results
            
        except Exception as e:
            logger.warning(f"Fallback ranking failed: {e}")
            return []
    
    def _calculate_confidence(self, entities: List[MedicalEntity], umls_results: List[UMLSResult]) -> float:
        """Calculate overall confidence score for medical processing."""
        if not entities and not umls_results:
            return 0.0
        
        # Base confidence from entity extraction
        entity_confidence = min(len(entities) * 0.2, 0.6) if entities else 0.0
        
        # Additional confidence from UMLS results
        umls_confidence = min(len(umls_results) * 0.15, 0.4) if umls_results else 0.0
        
        total_confidence = entity_confidence + umls_confidence
        return min(total_confidence, 1.0)
    
    def format_medical_context(self, medical_context: MedicalContext) -> str:
        """Format medical context for RAG integration."""
        if not medical_context.umls_results:
            return ""
        
        context_parts = []
        
        # Add medical entities information
        if medical_context.medical_entities:
            entities_text = ", ".join([e.term for e in medical_context.medical_entities])
            context_parts.append(f"Medical entities: {entities_text}")
        
        # Add UMLS knowledge
        for result in medical_context.umls_results:
            if result.relation_label and result.related_concept:
                context_parts.append(
                    f"{result.name} {result.relation_label} {result.related_concept}"
                )
        
        return " | ".join(context_parts)
    
    def get_medical_metadata(self, medical_context: MedicalContext) -> Dict[str, Any]:
        """Get medical metadata for RAG response."""
        return {
            "medical_entities": [
                {
                    "term": e.term,
                    "cui": e.cui,
                    "name": e.name,
                    "confidence": e.confidence
                }
                for e in medical_context.medical_entities
            ],
            "umls_results": [
                {
                    "cui": r.cui,
                    "name": r.name,
                    "relation": r.relation_label,
                    "related_concept": r.related_concept,
                    "score": r.score
                }
                for r in medical_context.umls_results
            ],
            "confidence_score": medical_context.confidence_score,
            "processing_time": medical_context.processing_time,
            "translation_used": medical_context.original_query != medical_context.translated_query
        }
    
    def is_medical_query(self, query: str, threshold: float = 0.3) -> bool:
        """Determine if a query is medical-related."""
        try:
            # Quick check using medical NER
            if self.medical_ner:
                entities = self.medical_ner.predict(query, min_score=0.5)
                return len(entities) > 0
            
            # Fallback: simple keyword matching
            medical_keywords = [
                "bệnh", "thuốc", "triệu chứng", "điều trị", "chẩn đoán", "y tế", "sức khỏe",
                "disease", "medicine", "symptom", "treatment", "diagnosis", "medical", "health",
                "patient", "doctor", "hospital", "clinic", "therapy"
            ]
            
            query_lower = query.lower()
            return any(keyword in query_lower for keyword in medical_keywords)
            
        except Exception as e:
            logger.warning(f"Medical query detection failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "translation_enabled": self.enable_translation,
            "medical_ner_enabled": self.enable_medical_ner,
            "umls_enabled": self.enable_umls,
            "max_umls_results": self.max_umls_results,
            "components_status": {
                "translator": self.translator is not None,
                "medical_ner": self.medical_ner is not None,
                "umls_api": self.umls_api is not None,
                "umlsbert": getattr(self, 'umlsbert', None) is not None
            }
        } 