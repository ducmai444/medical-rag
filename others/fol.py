from typing import List, Dict, Any
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define logical relation groups
LOGIC_GROUPS = {
    "Hierarchy": [
        "isa", "inverse_isa", "part_of", "has_part", "classified_as", "classifies",
        "constitutional_part_of", "conceptual_part_of", "regional_part_of", "has_regional_part",
        "system_of", "has_system", "supersystem_of", "has_supersystem", "entire_anatomy_structure_of",
        "has_entire_anatomy_structure", "anatomic_structure_is_physical_part_of",
        "has_physical_part_of_anatomic_structure", "develops_from", "develops_into",
        "matures_from", "matures_into", "has_developmental_stage", "developmental_stage_of",
        "has_structural_class", "structural_class_of", "has_chemical_classification",
        "is_chemical_classification_of_gene_product", "has_class", "class_of",
        "has_subtype", "is_subtype_of", "has_parent", "parent_of", "has_member", "member_of",
        "pathway_has_gene_element"
    ],
    "Treatment": [
        "treats", "treated_by", "may_be_treated_by", "has_therapeutic_class", "therapeutic_class_of",
        "regimen_has_accepted_use_for_disease", "disease_has_accepted_treatment_with_regimen",
        "associated_procedure_of", "has_associated_procedure", "procedure_has_target_anatomy",
        "procedure_has_completely_excised_anatomy", "procedure_has_partially_excised_anatomy",
        "procedure_has_excised_anatomy", "procedure_has_imaged_anatomy", "excised_anatomy_has_procedure",
        "imaged_anatomy_has_procedure", "has_direct_device", "direct_device_of", "has_procedure_device",
        "procedure_device_of", "uses_device", "device_used_by", "has_surgical_approach",
        "surgical_approach_of", "has_technique", "technique_of", "has_instrumentation",
        "instrumentation_of", "chemotherapy_regimen_has_component", "is_component_of_chemotherapy_regimen",
        "has_dose_form", "dose_form_of", "has_pharmaceutical_route", "pharmaceutical_route_of"
    ],
    "Causation": [
        "causes", "induced_by", "causative_agent_of", "has_causative_agent", "due_to", "result_of",
        "chemical_or_drug_initiates_biological_process", "biological_process_has_initiator_chemical_or_drug",
        "gene_involved_in_pathogenesis_of_disease", "pathogenesis_of_disease_involves_gene",
        "has_pathological_process", "pathological_process_of", "has_result", "result_of",
        "biological_process_has_result_biological_process", "biological_process_has_result_chemical_or_drug",
        "biological_process_results_from_biological_process", "biological_process_has_result_anatomy",
        "has_cause", "is_cause_of", "has_related_factor", "related_factor_of"
    ],
    "Symptom": [
        "has_symptom", "sign_or_symptom_of", "disease_has_finding", "has_finding", "manifestation_of",
        "has_manifestation", "has_definitional_manifestation", "has_associated_morphology",
        "associated_morphology_of", "has_direct_morphology", "direct_morphology_of", "has_clinical_course",
        "clinical_course_of", "has_severity", "severity_of", "has_phenotype", "phenotype_of",
        "has_property", "property_of", "has_observed_anatomical_entity", "anatomical_entity_observed_in",
        "has_associated_observation", "associated_observation_of"
    ],
    "Interaction": [
        "interacts_with", "associated_with", "co-occurs_with", "clinically_associated_with",
        "has_associated_disease", "disease_has_associated_disease", "has_associated_finding",
        "associated_finding_of", "chemical_or_drug_affects_gene_product", "gene_product_affected_by_chemical_or_drug",
        "chemical_or_drug_affects_cell_type_or_tissue", "cell_type_or_tissue_affected_by_chemical_or_drug",
        "biological_process_involves_chemical_or_drug", "biological_process_involves_gene_product",
        "gene_product_is_element_in_pathway", "gene_is_element_in_pathway",
        "has_associated_function", "associated_function_of", "has_associated_etiologic_finding",
        "associated_etiologic_finding_of", "has_associated_observation", "associated_observation_of",
        "is_associated_anatomic_site_of", "has_associated_condition"
    ],
    "Mapping_Identity": [
        "same_as", "mapped_to", "mapped_from", "translation_of", "has_translation",
        "transliterated_form_of", "has_transliterated_form", "has_permuted_term", "permuted_term_of",
        "corresponds_to", "primary_mapped_to", "primary_mapped_from", "uniquely_mapped_to",
        "uniquely_mapped_from", "multiply_mapped_to", "multiply_mapped_from", "has_form", "form_of",
        "has_expanded_form", "expanded_form_of", "has_british_form", "british_form_of",
        "has_clinician_form", "clinician_form_of", "has_common_name", "common_name_of",
        "has_alias", "alias_of", "has_entry_term", "entry_term_of",
        "version_of", "has_version"
    ]
}

class FOLReasoner:
    """
    First Order Logic (FOL) Reasoner for medical knowledge graphs.
    
    This class implements various logical rules to infer new relationships
    between medical concepts based on existing knowledge graph triplets.
    """
    
    def __init__(self):
        """Initialize the FOL Reasoner."""
        self.diagnosis_relations = ["diagnoses", "diagnosed_by"]
        
    @staticmethod
    def remove_duplicate(kg_triples: List[Any]) -> List[Any]:
        """
        Remove duplicate triplets from the knowledge graph.
        
        Args:
            kg_triples (List[Any]): List of knowledge graph triplets
            
        Returns:
            List[Any]: Deduplicated list of triplets
        """
        seen = set()
        result = []
        for item in kg_triples:
            if isinstance(item, dict):
                # For dictionaries, create a tuple of sorted items
                item_tuple = tuple(sorted(item.items()))
            elif isinstance(item, list):
                # For lists, convert to tuple
                item_tuple = tuple(item)
            else:
                # For other types, use as is
                item_tuple = item
                
            if item_tuple not in seen:
                seen.add(item_tuple)
                result.append(item)
        
        return result

    def apply_rules_to_kg(self, kg_triplets):
        """
        Apply logical rules to knowledge graph triplets to infer new relationships.
        
        Args:
            kg_triplets (List[Dict[str, str]]): List of knowledge graph triplets
            
        Returns:
            List[List[str]]: List of inferred relationship triplets
        """
        try:
            # Remove duplicates from input
            kg_triplets = self.remove_duplicate(kg_triplets)
            
            # Initialize containers for different types of inferred relations
            inferred_relations = defaultdict(list)
            
            # Apply each rule
            self._apply_co_occurrence_rule(kg_triplets, inferred_relations)
            self._apply_prevention_rule(kg_triplets, inferred_relations)
            self._apply_treatment_rule(kg_triplets, inferred_relations)
            self._apply_diagnosis_rule(kg_triplets, inferred_relations)
            self._apply_conjunction_rule(kg_triplets, inferred_relations)
            
            # Combine and deduplicate all inferred relations
            result = []
            for relation_type in inferred_relations.values():
                result.extend(self.remove_duplicate(relation_type))
                
            return self.remove_duplicate(result)
            
        except Exception as e:
            logger.error("Error applying FOL rules: %s", str(e))
            return []

    def _apply_co_occurrence_rule(self, kg_triplets, 
                                inferred_relations):
        """Apply the rule of co-occurrence."""
        for triplet in kg_triplets:
            if triplet.get("additionalRelationLabel") in LOGIC_GROUPS['Interaction']:
                for other_triplet in kg_triplets:
                    if (other_triplet.get("relatedFromIdName") == triplet.get("relatedIdName") and 
                        other_triplet.get("additionalRelationLabel") in LOGIC_GROUPS["Causation"]):
                        if triplet.get("relatedFromIdName") != other_triplet.get("relatedIdName"):
                            inferred_relations['co_occurs'].append([
                                triplet.get("relatedFromIdName"),
                                "affects",
                                other_triplet.get("relatedIdName")
                            ])

    def _apply_prevention_rule(self, kg_triplets, 
                             inferred_relations):
        """Apply the rule of prevention and causation."""
        for triplet in kg_triplets:
            if triplet.get("additionalRelationLabel") in LOGIC_GROUPS['Treatment']:
                for other_triplet in kg_triplets:
                    if (other_triplet.get("relatedFromIdName") == triplet.get("relatedIdName") and 
                        other_triplet.get("additionalRelationLabel") in LOGIC_GROUPS['Causation']):
                        if triplet.get("relatedFromIdName") != other_triplet.get("relatedIdName"):
                            inferred_relations['prevents'].append([
                                triplet.get("relatedFromIdName"),
                                "prevents",
                                other_triplet.get("relatedIdName")
                            ])

    def _apply_treatment_rule(self, kg_triplets, 
                            inferred_relations):
        """Apply the rule of treatment and classification."""
        for triplet in kg_triplets:
            if triplet.get("additionalRelationLabel") in LOGIC_GROUPS["Treatment"]:
                for other_triplet in kg_triplets:
                    if (other_triplet.get("relatedFromIdName") == triplet.get("relatedIdName") and 
                        other_triplet.get("additionalRelationLabel") in LOGIC_GROUPS["Hierarchy"]):
                        if triplet.get("relatedFromIdName") != other_triplet.get("relatedIdName"):
                            inferred_relations['treats'].append([
                                triplet.get("relatedFromIdName"),
                                "treats",
                                other_triplet.get("relatedIdName")
                            ])

    def _apply_diagnosis_rule(self, kg_triplets, 
                            inferred_relations):
        """Apply the rule of diagnosis and interaction."""
        for triplet in kg_triplets:
            if triplet.get("additionalRelationLabel") in self.diagnosis_relations:
                for other_triplet in kg_triplets:
                    if (other_triplet.get("relatedFromIdName") == triplet.get("relatedFromIdName") and 
                        other_triplet.get("additionalRelationLabel") in LOGIC_GROUPS["Interaction"]):
                        if other_triplet.get("relatedIdName") != triplet.get("relatedFromIdName"):
                            inferred_relations['diagnoses'].append([
                                other_triplet.get("relatedIdName"),
                                "diagnoses",
                                triplet.get("relatedIdName")
                            ])

    def _apply_conjunction_rule(self, kg_triplets, 
                              inferred_relations):
        """Apply the rule of conjunction."""
        for triplet in kg_triplets:
            if triplet.get("additionalRelationLabel") in LOGIC_GROUPS["Interaction"]:
                for other_triplet in kg_triplets:
                    if (other_triplet.get("relatedFromIdName") == triplet.get("relatedFromIdName") and 
                        other_triplet.get("additionalRelationLabel") in LOGIC_GROUPS["Causation"]):
                        if triplet.get("relatedIdName") != other_triplet.get("relatedIdName"):
                            inferred_relations['conjunction'].append([
                                triplet.get("relatedIdName"),
                                "co-occurs_with",
                                other_triplet.get("relatedIdName")
                            ])

# Example usage:
# reasoner = FOLReasoner()
# inferred_relations = reasoner.apply_rules_to_kg(knowledge_graph_triplets)