"""
Medical Knowledge Pipeline for RAG system.
Integrates Vietnamese translation, medical NER, UMLS API, MMR ranking, and Cross Encoder reranking.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import time
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

# Import medical modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'others'))

from translation import EnViT5Translator
from ner import MedicalNERLLM
from umls import UMLS_API
from ranking import MMR_ranking, similarity_score
from fol import FOLReasoner

logger = logging.getLogger(__name__)

# Medical terms dictionary from notebook
MEDICAL_TERMS = {
  # Drugs (generic and brand)
  "amoxicillin": ("C0002637", "Amoxicillin", "A broad-spectrum penicillin antibiotic used to treat various infections."),
  "ceftaroline": ("C0564627", "Ceftaroline", "A cephalosporin antibiotic effective against MRSA, used for skin and respiratory infections:contentReference[oaicite:8]{index=8}."),
  "doxycycline": ("C0004057", "Doxycycline", "A tetracycline antibiotic used for infections including acne, malaria prevention, and Lyme disease."),
  "vancomycin": ("C0040506", "Vancomycin", "A glycopeptide antibiotic used to treat serious gram-positive infections."),
  "azithromycin": ("C0003535", "Azithromycin", "A macrolide antibiotic used for respiratory, skin, and sexually transmitted infections."),
  "levofloxacin": ("C0149689", "Levofloxacin", "A fluoroquinolone antibiotic for various bacterial infections."),
  "fidaxomicin": ("C2821202", "Fidaxomicin", "A macrolide antibiotic primarily used to treat Clostridioides difficile colitis."),
  "dalbavancin": ("C2820913", "Dalbavancin", "A long-acting lipoglycopeptide antibiotic for acute bacterial skin infections."),
  "oritavancin": ("C2821380", "Oritavancin", "A lipoglycopeptide antibiotic used to treat Gram-positive skin infections."),
  "delafloxacin": ("C1284983", "Delafloxacin", "A fluoroquinolone antibiotic for acute bacterial skin and lung infections."),
  "tigecycline": ("C1308565", "Tigecycline", "A glycylcycline antibiotic effective against multi-drug-resistant organisms."),
  "ceftolozane": ("C0790322", "Ceftolozane", "A cephalosporin antibiotic often combined with tazobactam for resistant infections."),
  "ceftolozane/tazobactam": ("C1533118", "Ceftolozane-Tazobactam", "A combination antibiotic for complicated intra-abdominal and urinary tract infections."),
  
  # Antivirals and antimalarials
  "remdesivir": ("C1474153", "Remdesivir", "An antiviral drug approved for COVID-19, targeting viral RNA polymerase."),
  "molnupiravir": ("C3263273", "Molnupiravir", "An oral antiviral for COVID-19 that introduces copying errors during viral replication."),
  "nirmatrelvir": ("C3545470", "Nirmatrelvir", "A SARS-CoV-2 protease inhibitor used in combination therapy for COVID-19 (Paxlovid)."),
  "baloxavir": ("C4062026", "Baloxavir", "An antiviral approved for influenza, targeting the viral cap-dependent endonuclease."),
  "ivermectin": ("C0021485", "Ivermectin", "An antiparasitic medication used for onchocerciasis and strongyloidiasis."),
  "artesunate": ("C3462231", "Artesunate", "An artemisinin derivative used to treat severe malaria."),
  "tafenoquine": ("C3271764", "Tafenoquine", "An antimalarial drug used for relapse prevention in vivax malaria."),
  
  # Cancer therapies
  "ibritumomab tiuxetan": ("C0279390", "Ibritumomab Tiuxetan", "A radioimmunotherapy targeting CD20 on B-cells (Zevalin) for certain lymphomas."),
  "nivolumab": ("C1851170", "Nivolumab", "A PD-1 immune checkpoint inhibitor antibody used to treat various advanced cancers:contentReference[oaicite:9]{index=9}."),
  "pembrolizumab": ("C1855280", "Pembrolizumab", "A PD-1 immune checkpoint inhibitor antibody used for melanoma, lung cancer, and others."),
  "atezolizumab": ("C1622437", "Atezolizumab", "A PD-L1 immune checkpoint inhibitor antibody used for urothelial and lung cancers."),
  "durvalumab": ("C1604100", "Durvalumab", "A PD-L1 immune checkpoint inhibitor antibody used for bladder and lung cancers."),
  "avelumab": ("C1622366", "Avelumab", "A PD-L1 immune checkpoint inhibitor antibody used for Merkel cell carcinoma and others."),
  "ipilimumab": ("C1620927", "Ipilimumab", "A CTLA-4 immune checkpoint inhibitor antibody used for melanoma and renal carcinoma."),
  "ibrutinib": ("C1574738", "Ibrutinib", "A Bruton's tyrosine kinase inhibitor used in chronic lymphocytic leukemia and lymphoma."),
  "acalabrutinib": ("C1575185", "Acalabrutinib", "A second-generation BTK inhibitor for CLL and mantle cell lymphoma."),
  "lenalidomide": ("C1547141", "Lenalidomide", "An immunomodulatory drug used in multiple myeloma and myelodysplastic syndromes."),
  "pomalidomide": ("C1566210", "Pomalidomide", "An immunomodulatory drug for relapsed or refractory multiple myeloma."),
  "carfilzomib": ("C1543420", "Carfilzomib", "A proteasome inhibitor used in relapsed multiple myeloma."),
  "bortezomib": ("C0013448", "Bortezomib", "A proteasome inhibitor used in multiple myeloma and mantle cell lymphoma."),
  "blinatumomab": ("C1567949", "Blinatumomab", "A bispecific T-cell engager (CD19) used for acute lymphoblastic leukemia."),
  "tisagenlecleucel": ("C1706443", "Tisagenlecleucel", "A CD19-directed CAR T-cell therapy for refractory leukemia/lymphoma."),
  "axicabtagene ciloleucel": ("C1706475", "Axicabtagene Ciloleucel", "A CAR T-cell therapy for large B-cell lymphoma."),
  "palbociclib": ("C1567321", "Palbociclib", "A CDK4/6 inhibitor used in ER-positive, HER2-negative breast cancer."),
  "ribociclib": ("C1579765", "Ribociclib", "A CDK4/6 inhibitor for hormone receptor-positive breast cancer."),
  "abemaciclib": ("C1556827", "Abemaciclib", "A CDK4/6 inhibitor used in metastatic breast cancer."),
  "olaparib": ("C1566249", "Olaparib", "A PARP inhibitor for ovarian and breast cancer with BRCA mutation."),
  "niraparib": ("C1557761", "Niraparib", "A PARP inhibitor for ovarian cancer maintenance therapy."),
  "rucaparib": ("C1554562", "Rucaparib", "A PARP inhibitor for BRCA-mutated ovarian cancer."),
  "lenvatinib": ("C1576431", "Lenvatinib", "A multikinase inhibitor used in thyroid carcinoma and hepatocellular carcinoma."),
  "sorafenib": ("C1349073", "Sorafenib", "A multikinase inhibitor used for renal and liver cancers."),
  "sunitinib": ("C1337139", "Sunitinib", "A tyrosine kinase inhibitor used for renal cell carcinoma and GIST."),
  "regorafenib": ("C1578946", "Regorafenib", "A multi-kinase inhibitor used in colorectal cancer and GIST."),
  "cabozantinib": ("C1569040", "Cabozantinib", "A tyrosine kinase inhibitor for medullary thyroid and renal carcinomas."),
  
  # Biologics / Immunomodulators
  "adalimumab": ("C0021203", "Adalimumab", "A TNF-alpha inhibitor monoclonal antibody used in rheumatoid arthritis and psoriasis."),
  "infliximab": ("C0021217", "Infliximab", "A TNF-alpha inhibitor monoclonal antibody for Crohn's disease and arthritis."),
  "etanercept": ("C0021309", "Etanercept", "A TNF receptor fusion protein used in autoimmune inflammatory diseases."),
  "ustekinumab": ("C1578473", "Ustekinumab", "A monoclonal antibody against IL-12/23 (p40) used for psoriasis and psoriatic arthritis."),
  "secukinumab": ("C1576773", "Secukinumab", "An IL-17A inhibitor monoclonal antibody for psoriasis and ankylosing spondylitis."),
  "ixekizumab": ("C1555670", "Ixekizumab", "An IL-17A inhibitor monoclonal antibody used in psoriasis and psoriatic arthritis."),
  "brodalumab": ("C1540648", "Brodalumab", "An IL-17 receptor A inhibitor for severe plaque psoriasis."),
  "dupilumab": ("C1558429", "Dupilumab", "An IL-4 receptor alpha antagonist monoclonal antibody for asthma and eczema."),
  "tildrakizumab": ("C1701713", "Tildrakizumab", "An IL-23 inhibitor monoclonal antibody for psoriasis."),
  "risankizumab": ("C1701715", "Risankizumab", "An IL-23 inhibitor monoclonal antibody for plaque psoriasis."),
  "tocilizumab": ("C0034061", "Tocilizumab", "An IL-6 receptor antagonist monoclonal antibody used for rheumatoid arthritis."),
  "sarilumab": ("C1576879", "Sarilumab", "An IL-6 receptor antagonist monoclonal antibody for rheumatoid arthritis."),
  "anakinra": ("C0024085", "Anakinra", "An IL-1 receptor antagonist used for rheumatoid arthritis and autoinflammatory syndromes."),
  "belimumab": ("C1551497", "Belimumab", "A monoclonal antibody targeting B-lymphocyte stimulator (BLyS) for systemic lupus erythematosus."),
  "rituximab": ("C0035813", "Rituximab", "An anti-CD20 monoclonal antibody used in B-cell lymphomas and autoimmune diseases."),
  "vedolizumab": ("C1548296", "Vedolizumab", "An integrin inhibitor monoclonal antibody for inflammatory bowel disease."),
  "tofacitinib": ("C3554708", "Tofacitinib", "A JAK inhibitor used to treat rheumatoid arthritis and other autoimmune conditions:contentReference[oaicite:10]{index=10}."),
  "baricitinib": ("C3557265", "Baricitinib", "A JAK inhibitor for rheumatoid arthritis and recently for COVID-19."),
  "upadacitinib": ("C3558577", "Upadacitinib", "A selective JAK1 inhibitor used for rheumatoid arthritis."),
  "tepoxalin": ("C1544755", "Tepoxalin", "A dual COX/5-LOX inhibitor used in veterinary medicine (for context of rare NSAIDs)."),
  
  # Psychiatric medications
  "risperidone": ("C0005584", "Risperidone", "An atypical antipsychotic used in schizophrenia and bipolar disorder."),
  "brexpiprazole": ("C1557767", "Brexpiprazole", "A serotonin–dopamine activity modulator for schizophrenia and major depression."),
  "lurasidone": ("C1572389", "Lurasidone", "An atypical antipsychotic for schizophrenia and bipolar depression."),
  "asenapine": ("C1558209", "Asenapine", "An atypical antipsychotic administered sublingually, used for schizophrenia and bipolar disorder."),
  "esketamine": ("C1701807", "Esketamine", "An NMDA receptor antagonist nasal spray for treatment-resistant depression."),
  "cariprazine": ("C1541378", "Cariprazine", "An atypical antipsychotic for schizophrenia and bipolar mania."),
  
  # Endocrine and metabolic drugs
  "dapagliflozin": ("C1577772", "Dapagliflozin", "An SGLT2 inhibitor for type 2 diabetes and heart failure."),
  "empagliflozin": ("C1577771", "Empagliflozin", "An SGLT2 inhibitor for type 2 diabetes and heart failure."),
  "canagliflozin": ("C1577769", "Canagliflozin", "An SGLT2 inhibitor for type 2 diabetes, reduces cardiovascular risk."),
  "liraglutide": ("C1557395", "Liraglutide", "A GLP-1 receptor agonist for type 2 diabetes and obesity."),
  "semaglutide": ("C1557404", "Semaglutide", "A GLP-1 receptor agonist for type 2 diabetes and obesity."),
  "exenatide": ("C1546440", "Exenatide", "A GLP-1 receptor agonist for type 2 diabetes."),
  "levothyroxine": ("C0027356", "Levothyroxine", "A synthetic thyroid hormone used to treat hypothyroidism."),
  
  # Cardiovascular drugs
  "alirocumab": ("C1578504", "Alirocumab", "A PCSK9 inhibitor monoclonal antibody for lowering LDL cholesterol."),
  "evolocumab": ("C1578506", "Evolocumab", "A PCSK9 inhibitor monoclonal antibody to reduce cholesterol levels."),
  "apixaban": ("C1558944", "Apixaban", "An oral factor Xa inhibitor anticoagulant for atrial fibrillation and VTE."),
  "rivaroxaban": ("C1558936", "Rivaroxaban", "An oral factor Xa inhibitor anticoagulant for thrombosis and stroke prevention."),
  "dabigatran": ("C1559003", "Dabigatran", "An oral direct thrombin inhibitor anticoagulant."),
  "bempedoic acid": ("C1557861", "Bempedoic Acid", "An ATP-citrate lyase inhibitor to lower LDL cholesterol."),
  
  # Other notable drugs
  "linezolid": ("C0024174", "Linezolid", "An oxazolidinone antibiotic for Gram-positive infections."),
  "metronidazole": ("C0026184", "Metronidazole", "An antibiotic effective against anaerobic bacteria and protozoa."),
  "trimethoprim/sulfamethoxazole": ("C0024749", "Trimethoprim-Sulfamethoxazole", "A combination antibiotic for UTIs, pneumonia (Pneumocystis) and other infections."),
  "isoniazid": ("C0020839", "Isoniazid", "First-line antitubercular medication."),
  "rifampin": ("C0036051", "Rifampin", "A rifamycin antibiotic used in tuberculosis treatment."),
  "ethambutol": ("C0002963", "Ethambutol", "An antimycobacterial agent used in tuberculosis therapy."),
  "pyrazinamide": ("C0034149", "Pyrazinamide", "An antitubercular drug used with isoniazid and rifampin."),
  
  # Diseases and conditions
  "systemic lupus erythematosus": ("C0024141", "Systemic Lupus Erythematosus", "An autoimmune disease in which the immune system attacks multiple organs:contentReference[oaicite:11]{index=11}."),
  "rheumatoid arthritis": ("C0003873", "Rheumatoid Arthritis", "A chronic autoimmune disorder causing inflammation of joints:contentReference[oaicite:12]{index=12}."),
  "multiple sclerosis": ("C0026769", "Multiple Sclerosis", "A demyelinating disease of the central nervous system:contentReference[oaicite:13]{index=13}."),
  "amyotrophic lateral sclerosis": ("C0002736", "Amyotrophic Lateral Sclerosis", "A progressive neurodegenerative disease affecting motor neurons:contentReference[oaicite:14]{index=14}."),
  "guillain-barre syndrome": ("C0019321", "Guillain-Barr Syndrome", "An acute autoimmune neuropathy causing rapid muscle weakness:contentReference[oaicite:15]{index=15}."),
  "myasthenia gravis": ("C0027051", "Myasthenia Gravis", "A chronic autoimmune neuromuscular disease characterized by muscle weakness."),
  "huntington disease": ("C0020179", "Huntington Disease", "A genetic neurodegenerative disorder causing movement, cognitive, and psychiatric disturbances."),
  "parkinson disease": ("C0030567", "Parkinson Disease", "A neurodegenerative disorder characterized by tremor, rigidity, and bradykinesia."),
  "alzheimer disease": ("C0002395", "Alzheimer Disease", "A neurodegenerative disease causing dementia and cognitive decline."),
  "sarcoidosis": ("C0033866", "Sarcoidosis", "An inflammatory disease characterized by granulomas in multiple organs, especially lungs."),
  "systemic sclerosis": ("C0036161", "Systemic Sclerosis", "An autoimmune connective tissue disease causing skin and organ fibrosis."),
  "hashimoto thyroiditis": ("C0021642", "Hashimoto Thyroiditis", "An autoimmune thyroid disorder causing hypothyroidism."),
  "graves disease": ("C0017725", "Graves Disease", "An autoimmune disorder causing hyperthyroidism and goiter."),
  "addison disease": ("C0002651", "Addison Disease", "Primary adrenal insufficiency, often autoimmune, causing cortisol deficiency."),
  "cushing syndrome": ("C0007873", "Cushing Syndrome", "A condition caused by chronic high cortisol, often from steroids or adrenal tumor."),
  "diabetes mellitus type 1": ("C0011860", "Diabetes Mellitus Type 1", "An autoimmune destruction of pancreatic beta cells leading to insulin deficiency."),
  "diabetes mellitus type 2": ("C0011860", "Diabetes Mellitus Type 2", "A metabolic disorder characterized by insulin resistance and relative insulin deficiency."),
  "polycystic ovary syndrome": ("C0030305", "Polycystic Ovary Syndrome", "An endocrine disorder causing ovulatory dysfunction and hyperandrogenism."),
  "cystic fibrosis": ("C0010674", "Cystic Fibrosis", "A genetic disorder affecting chloride channels, causing thick mucus in lungs and GI tract."),
  "sickle cell anemia": ("C0023433", "Sickle Cell Anemia", "A hereditary hemoglobinopathy causing sickle-shaped red blood cells and vaso-occlusion."),
  "beta thalassemia": ("C0005842", "Beta Thalassemia", "A genetic disorder causing reduced beta-globin production and anemia."),
  "gauchers disease": ("C0027817", "Gaucher Disease", "A lysosomal storage disorder caused by glucocerebrosidase deficiency."),
  "tay sachs disease": ("C0032451", "Tay-Sachs Disease", "A lysosomal storage disorder caused by hexosaminidase A deficiency, leading to neurodegeneration."),
  "pompe disease": ("C0029363", "Pompe Disease", "A glycogen storage disorder (acid maltase deficiency) affecting heart and muscles."),
  "fabry disease": ("C0016167", "Fabry Disease", "A lysosomal storage disorder caused by alpha-galactosidase A deficiency."),
  "metachromatic leukodystrophy": ("C0025595", "Metachromatic Leukodystrophy", "A genetic disorder causing myelin sheath degeneration in nerves."),
  "amyloidosis": ("C0004491", "Amyloidosis", "A group of conditions where misfolded proteins deposit as amyloid in organs."),
  "hemochromatosis": ("C1386814", "Hemochromatosis", "An iron overload disorder that can damage liver, heart, and pancreas."),
  "wilsons disease": ("C0042377", "Wilson Disease", "A genetic disorder causing copper accumulation, leading to liver and neurological disease."),
  "phenylketonuria": ("C0037356", "Phenylketonuria", "An inherited metabolic disorder causing phenylalanine accumulation."),
  "maple syrup urine disease": ("C0521488", "Maple Syrup Urine Disease", "An inherited disorder causing branched-chain amino acid accumulation."),
  "mucopolysaccharidosis": ("C0430027", "Mucopolysaccharidosis", "A group of inherited metabolic disorders affecting glycosaminoglycan breakdown."),
  
  # Infectious diseases
  "tuberculosis": ("C0041296", "Tuberculosis", "A chronic infectious disease caused by Mycobacterium tuberculosis, usually affecting lungs."),
  "leprosy": ("C0024109", "Leprosy", "A chronic infection by Mycobacterium leprae affecting skin and nerves."),
  "dengue fever": ("C0019221", "Dengue Fever", "A mosquito-borne viral infection causing fever, rash, and severe joint pain."),
  "ebola hemorrhagic fever": ("C0019041", "Ebola Virus Disease", "A severe viral hemorrhagic fever with high mortality."),
  "sars": ("C0036790", "Severe Acute Respiratory Syndrome", "A viral respiratory illness caused by a coronavirus, first identified in 2003."),
  "mers": ("C3534218", "Middle East Respiratory Syndrome", "A viral respiratory disease caused by the MERS coronavirus."),
  "lyme disease": ("C0024651", "Lyme Disease", "An infectious disease caused by Borrelia burgdorferi, transmitted by ticks."),
  "zika virus infection": ("C3547502", "Zika Virus Infection", "A mosquito-borne viral disease that can cause birth defects."),
  "chikungunya": ("C0008034", "Chikungunya", "A mosquito-borne viral disease causing fever and joint pain."),
  "malaria": ("C0025289", "Malaria", "A mosquito-borne parasitic infection causing cyclical fevers and anemia."),
  "influenza": ("C0021400", "Influenza", "An acute respiratory viral infection caused by influenza viruses."),
  
  # Cancers (disease names)
  "acute lymphoblastic leukemia": ("C0005612", "Acute Lymphoblastic Leukemia", "A rapidly progressing cancer of lymphoid lineage, common in children."),
  "acute myeloid leukemia": ("C0022694", "Acute Myeloid Leukemia", "A rapidly progressing cancer of myeloid blood cells."),
  "chronic lymphocytic leukemia": ("C0007797", "Chronic Lymphocytic Leukemia", "A slow-growing cancer of B lymphocytes in adults."),
  "chronic myeloid leukemia": ("C0007786", "Chronic Myeloid Leukemia", "A myeloproliferative neoplasm associated with the BCR-ABL fusion gene."),
  "multiple myeloma": ("C0002645", "Multiple Myeloma", "A malignant proliferation of plasma cells in the bone marrow."),
  "hodgkin lymphoma": ("C0019204", "Hodgkin Lymphoma", "A cancer of the lymphatic system characterized by Reed-Sternberg cells."),
  "non-hodgkin lymphoma": ("C0024017", "Non-Hodgkin Lymphoma", "A diverse group of lymphoid cancers without Reed-Sternberg cells."),
  "melanoma": ("C0025202", "Melanoma", "A malignant skin tumor arising from melanocytes."),
  "breast cancer": ("C0006142", "Breast Cancer", "A malignant tumor of breast tissue, often adenocarcinoma."),
  "prostate cancer": ("C0033578", "Prostate Cancer", "A malignant tumor of the prostate gland."),
  "lung cancer": ("C0024121", "Lung Cancer", "A malignant lung tumor, commonly small cell or non-small cell carcinoma."),
  "colon cancer": ("C0007102", "Colorectal Carcinoma", "A malignant tumor of the colon or rectum."),
  "pancreatic cancer": ("C0039731", "Pancreatic Carcinoma", "A malignant tumor arising from the pancreatic exocrine cells."),
  "hepatocellular carcinoma": ("C0007109", "Hepatocellular Carcinoma", "A primary liver cancer arising from hepatocytes."),
  "glioblastoma": ("C0242339", "Glioblastoma", "A highly malignant primary brain tumor (astrocytoma)."),
  "melanoma, uveal": ("C0041431", "Uveal Melanoma", "A malignant melanoma of the eye's uveal tract."),
  "carcinoid tumor": ("C0007091", "Carcinoid Tumor", "A slow-growing neuroendocrine tumor often of the gastrointestinal tract or lungs."),
  
  # Other conditions
  "acute respiratory distress syndrome": ("C0003392", "Acute Respiratory Distress Syndrome", "A severe form of lung injury causing respiratory failure."),
  "acute kidney injury": ("C0002907", "Acute Kidney Injury", "A sudden decline in renal function, previously called acute renal failure."),
  "chronic kidney disease": ("C0010970", "Chronic Kidney Disease", "Long-term loss of kidney function leading to renal failure."),
  "metabolic syndrome": ("C0457199", "Metabolic Syndrome", "A cluster of conditions (hypertension, dyslipidemia, etc.) increasing cardiovascular risk."),
  "polymyalgia rheumatica": ("C0022810", "Polymyalgia Rheumatica", "An inflammatory syndrome causing muscle pain and stiffness in older adults."),
  "temporal arteritis": ("C0041832", "Temporal Arteritis", "An inflammatory disease of large blood vessels (giant cell arteritis) often causing headache."),
  "factor v leiden thrombophilia": ("C1866765", "Factor V Leiden Thrombophilia", "A genetic mutation causing hypercoagulability due to Factor V resistance."),
  "paroxysmal nocturnal hemoglobinuria": ("C0205383", "Paroxysmal Nocturnal Hemoglobinuria", "An acquired hematopoietic stem cell disorder causing hemolysis."),
  "antiphospholipid syndrome": ("C0021309", "Antiphospholipid Syndrome", "An autoimmune disorder causing thrombosis due to antibodies against phospholipids."),
  "hemophagocytic lymphohistiocytosis": ("C0027653", "Hemophagocytic Lymphohistiocytosis", "An aggressive immune activation syndrome causing fever and cytopenias."),
}


@dataclass
class MedicalTerm:
    """Represents a medical term with UMLS information."""
    term: str
    cui: str = ""
    name: str = ""
    definition: str = ""
    relations: List[Dict] = None
    ranked_relations: List[Dict] = None
    
    def __post_init__(self):
        if self.relations is None:
            self.relations = []
        if self.ranked_relations is None:
            self.ranked_relations = []

@dataclass
class MedicalContext:
    """Combined medical context for RAG."""
    original_query: str
    translated_query: str
    medical_terms: List[str]
    umls_results: Dict[str, MedicalTerm]
    final_relations: List[Tuple[str, str, str]]  # (subject, relation, object)
    processing_time: float

class MedicalRAGPipeline:
    """
    Comprehensive medical pipeline for Vietnamese medical Q&A following notebook workflow.
    """
    
    def __init__(self, 
                 umls_api_key: str,
                 enable_translation: bool = True,
                 enable_caching: bool = False,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 max_workers: int = 4):
        """
        Initialize medical pipeline components.
        
        Args:
            umls_api_key: API key for UMLS access
            enable_translation: Enable Vietnamese-English translation
            enable_caching: Enable Redis caching
            redis_host: Redis host for caching
            redis_port: Redis port for caching
            max_workers: Max workers for parallel processing
        """
        self.umls_api_key = umls_api_key
        self.enable_translation = enable_translation
        self.enable_caching = enable_caching
        self.max_workers = max_workers
        
        # Initialize caching if enabled
        if enable_caching:
            try:
                import redis
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
                logger.info("✅ Redis caching enabled")
            except ImportError:
                logger.warning("Redis not available, caching disabled")
                self.enable_caching = False
                self.redis_client = None
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, caching disabled")
                self.enable_caching = False
                self.redis_client = None
        else:
            self.redis_client = None
        
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
            else:
                self.translator = None
            
            # Medical NER component
            logger.info("Initializing Medical NER...")
            try:
                self.medical_ner = MedicalNERLLM()
                logger.info("✅ Medical NER initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Medical NER: {e}")
                raise
            
            # UMLS API component
            logger.info("Initializing UMLS API...")
            try:
                self.umls_api = UMLS_API(self.umls_api_key)
                logger.info("✅ UMLS API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize UMLS API: {e}")
                raise
            
            # FOL Reasoner (optional)
            logger.info("Initializing FOL Reasoner...")
            try:
                self.fol_reasoner = FOLReasoner()
                logger.info("✅ FOL Reasoner initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize FOL Reasoner: {e}")
                self.fol_reasoner = None
            
            logger.info("Medical pipeline initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize medical pipeline: {e}")
            raise
    
    def process_query(self, query: str) -> MedicalContext:
        """
        Process a medical query through the complete pipeline following notebook workflow.
        
        Args:
            query: Input query (Vietnamese or English)
            
        Returns:
            MedicalContext with all processed information
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {query}")
            
            # Step 1: Translation (Vietnamese -> English)
            translated_query = self._translate_query(query)
            logger.info(f"Translated query: {translated_query}")
            
            # Step 2: Medical NER to extract medical terms
            medical_terms = self._extract_medical_terms(translated_query)
            logger.info(f"Medical terms extracted: {medical_terms}")
            
            # Step 3: Process each term through UMLS API and MMR ranking
            umls_results, final_relations = self._process_umls_terms(translated_query, medical_terms)
            
            processing_time = time.time() - start_time
            
            return MedicalContext(
                original_query=query,
                translated_query=translated_query,
                medical_terms=medical_terms,
                umls_results=umls_results,
                final_relations=final_relations,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Failed to process medical query: {e}")
            # Return minimal context on error
            return MedicalContext(
                original_query=query,
                translated_query=query,
                medical_terms=[],
                umls_results={},
                final_relations=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time
            )
    
    def _translate_query(self, query: str) -> str:
        """Translate Vietnamese query to English if needed."""
        if not self.enable_translation or not self.translator:
            return query
        
        try:
            translated = self.translator.translate(query)
            # Remove prefix if present (from notebook: [4:])
            if translated.startswith("en: "):
                translated = translated[4:]
            return translated
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return query
    
    def _extract_medical_terms(self, query: str) -> List[str]:
        """Extract medical terms using NER."""
        try:
            medical_terms = self.medical_ner.predict(query)
            return medical_terms if medical_terms else []
        except Exception as e:
            logger.warning(f"Medical NER failed: {e}")
            return []
    
    def _process_umls_terms(self, query: str, medical_terms: List[str]) -> Tuple[Dict[str, MedicalTerm], List[Tuple[str, str, str]]]:
        """Process medical terms through UMLS API with caching and parallel processing."""
        umls_results = {}
        
        # Process terms in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(
                lambda term: self._process_single_term(term, query), 
                medical_terms
            ))
        
        # Collect results
        for result in results:
            if result:
                cui, medical_term = result
                umls_results[cui] = medical_term
        
        # Collect final relations from MMR ranked results
        all_ranked_relations = []
        for term in umls_results.values():
            if term.ranked_relations:
                all_ranked_relations.extend(term.ranked_relations)
        
        # Take top 10 relations (MMR already ranked them)
        top_relations = all_ranked_relations[:10]
        
        # Format as tuples (subject, relation, object)
        final_relations = []
        for rel in top_relations:
            subject = rel.get("relatedFromIdName", "")
            relation = rel.get("additionalRelationLabel", "").replace("_", " ")
            obj = rel.get("relatedIdName", "")
            
            if subject and relation and obj:
                final_relations.append((subject, relation, obj))
        
        logger.info(f"MMR ranking completed, selected {len(final_relations)} final relations from {len(all_ranked_relations)} total relations")
        
        return umls_results, final_relations
    
    def _process_single_term(self, term: str, query: str) -> Optional[Tuple[str, MedicalTerm]]:
        """Process a single medical term through UMLS API."""
        try:
            # Correct spelling
            corrected_term = self.medical_ner.correct_spelling(term)
            
            # Search CUI with caching
            cui_results = self._cached_search_cui(corrected_term)
            
            if not cui_results:
                # Fallback to medical terms dictionary
                if corrected_term.lower() in MEDICAL_TERMS:
                    cui, name, definition = MEDICAL_TERMS[corrected_term.lower()]
                    return cui, MedicalTerm(
                        term=corrected_term,
                        cui=cui,
                        name=name,
                        definition=definition,
                        relations=[],
                        ranked_relations=[]
                    )
                return None
            
            cui, name = cui_results[0]
            
            # Get definitions with caching
            definitions = self._cached_get_definitions(cui)
            definition = ""
            if definitions:
                for def_item in definitions:
                    source = def_item.get("rootSource", "")
                    if source in ["MSH", "NCI", "ICF", "CSP", "HPO"]:
                        definition = def_item.get("value", "")
                        break
            
            # Get relations with caching
            relations = self._cached_get_relations(cui)
            
            # MMR ranking
            ranked_relations = []
            if relations:
                try:
                    ranked_relations = MMR_ranking(query, relations, top_k=10)
                except Exception as e:
                    logger.warning(f"MMR ranking failed for {term}: {e}")
                    ranked_relations = relations[:10]  # Fallback to first 10
            
            medical_term = MedicalTerm(
                term=corrected_term,
                cui=cui,
                name=name,
                definition=definition,
                relations=relations,
                ranked_relations=ranked_relations
            )
            
            return cui, medical_term
            
        except Exception as e:
            logger.warning(f"Failed to process term {term}: {e}")
            return None
    
    def _cached_search_cui(self, term: str):
        """Search CUI with optional caching."""
        if self.enable_caching and self.redis_client:
            key = f"search_cui:{term}"
            try:
                result = self.redis_client.get(key)
                if result:
                    return json.loads(result)
                result = self.umls_api.search_cui(term)
                self.redis_client.set(key, json.dumps(result), ex=43200)  # 12 hours
                return result
            except Exception as e:
                logger.warning(f"Cache operation failed: {e}")
                return self.umls_api.search_cui(term)
        else:
            return self.umls_api.search_cui(term)
    
    def _cached_get_definitions(self, cui: str):
        """Get definitions with optional caching."""
        if self.enable_caching and self.redis_client:
            key = f"definitions:{cui}"
            try:
                result = self.redis_client.get(key)
                if result:
                    return json.loads(result)
                result = self.umls_api.get_definitions(cui)
                self.redis_client.set(key, json.dumps(result), ex=43200)  # 12 hours
                return result
            except Exception as e:
                logger.warning(f"Cache operation failed: {e}")
                return self.umls_api.get_definitions(cui)
        else:
            return self.umls_api.get_definitions(cui)
    
    def _cached_get_relations(self, cui: str):
        """Get relations with optional caching."""
        if self.enable_caching and self.redis_client:
            key = f"relations:{cui}"
            try:
                result = self.redis_client.get(key)
                if result:
                    return json.loads(result)
                result = self.umls_api.get_relations(cui)
                self.redis_client.set(key, json.dumps(result), ex=43200)  # 12 hours
                return result
            except Exception as e:
                logger.warning(f"Cache operation failed: {e}")
                return self.umls_api.get_relations(cui)
        else:
            return self.umls_api.get_relations(cui)
    
    # def _calculate_confidence(self, medical_terms: List[str], umls_results: Dict[str, MedicalTerm], final_relations: List[Tuple]) -> float:
    #     """Calculate overall confidence score."""
    #     if not medical_terms and not umls_results and not final_relations:
    #         return 0.0
        
    #     # Base confidence from medical terms extraction
    #     terms_confidence = min(len(medical_terms) * 0.2, 0.4) if medical_terms else 0.0
        
    #     # Additional confidence from UMLS results
    #     umls_confidence = min(len(umls_results) * 0.15, 0.3) if umls_results else 0.0
        
    #     # Additional confidence from final relations
    #     relations_confidence = min(len(final_relations) * 0.05, 0.3) if final_relations else 0.0
        
    #     total_confidence = terms_confidence + umls_confidence + relations_confidence
    #     return min(total_confidence, 1.0)
    
    def format_medical_context(self, medical_context: MedicalContext) -> str:
        """Format medical context for RAG integration."""
        context_lines = []
        
        # Add medical terms definitions
        for cui, term in medical_context.umls_results.items():
            if term.name and term.definition:
                context_lines.append(f"Name: {term.name}")
                context_lines.append(f"Definition: {term.definition}")
                context_lines.append("")  # Empty line for separation
        
        # Add final relations
        for subject, relation, obj in medical_context.final_relations:
            context_lines.append(f"({subject},{relation},{obj})")
        
        return "\n".join(context_lines) if context_lines else ""
    
    def get_medical_metadata(self, medical_context: MedicalContext) -> Dict[str, Any]:
        """Get medical metadata for RAG response."""
        return {
            "medical_terms": medical_context.medical_terms,
            "umls_terms": [
                {
                    "cui": term.cui,
                    "name": term.name,
                    "term": term.term,
                    "definition": term.definition,
                    "relations_count": len(term.relations),
                    "ranked_relations_count": len(term.ranked_relations)
                }
                for term in medical_context.umls_results.values()
            ],
            "final_relations": [
                {"subject": rel[0], "relation": rel[1], "object": rel[2]}
                for rel in medical_context.final_relations
            ],
            "processing_time": medical_context.processing_time,
            "translation_used": medical_context.original_query != medical_context.translated_query
        }
    
    def is_medical_query(self, query: str) -> bool:
        """Determine if a query is medical-related."""
        try:
            # Quick check using medical NER
            if self.medical_ner:
                entities = self.medical_ner.predict(query)
                return len(entities) > 0
            
            return False
            
        except Exception as e:
            logger.warning(f"Medical query detection failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "translation_enabled": self.enable_translation,
            "caching_enabled": self.enable_caching,
            "max_workers": self.max_workers,
            "components_status": {
                "translator": self.translator is not None,
                "medical_ner": self.medical_ner is not None,
                "umls_api": self.umls_api is not None,
                "fol_reasoner": self.fol_reasoner is not None,
                "redis_client": self.redis_client is not None
            }
        } 