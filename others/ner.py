import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch
import re
from fuzzywuzzy import process
from typing import List, Dict, Union
import os

# Load medical terms from JSON file
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

MEDICAL_TERMS = {
  # Drugs (generic and brand)
  "amoxicillin": ("C0002637", "Amoxicillin", "A broad-spectrum penicillin antibiotic used to treat various infections."),
  "ceftaroline": ("C0564627", "Ceftaroline", "A cephalosporin antibiotic effective against MRSA, used for skin and respiratory infections:contentReference[oaicite:8]{index=8}."),
}

class MedicalNERLLM:
    def __init__(self, model_name: str = "blaze999/Medical-NER", device: str = None):
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=0 if device == "cuda" else -1
        )
        
        self.important_tags = [
            "DISEASE_DISORDER",
            "MEDICATION",
            "SIGN_SYMPTOM",
            "THERAPEUTIC_PROCEDURE",
            "DIAGNOSTIC_PROCEDURE",
            "BIOLOGICAL_STRUCTURE"
        ]
        
        self.max_length = 512
        self.min_term_length = 3
    
    def correct_spelling(self, term: str, threshold: int = 95) -> str:
        term_lower = term.lower().strip()
        if term_lower in MEDICAL_TERMS:
            return term_lower
        matches = process.extractOne(term_lower, MEDICAL_TERMS.keys(), score_cutoff=threshold)
        if matches:
            matched_term, score = matches
            return matched_term
        return term

    def predict(self, prompt: str, min_score: float = 0.0) -> List[str]:
        if not prompt or not isinstance(prompt, str):
            return []

        # Truncate if needed
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        if len(tokens) > self.max_length:
            prompt = self.tokenizer.decode(tokens[:self.max_length-2], skip_special_tokens=True)

        try:
            entities = self.ner_pipeline(prompt)
        except Exception as e:
            print(f"Error in NER pipeline: {e}")
            entities = []

        medical_terms = []
        
        # Get terms from NER
        for entity in entities:
            if (entity["entity_group"] in self.important_tags and 
                entity.get("score", 1.0) >= min_score):
                medical_terms.append(entity["word"])

        # Fallback to dictionary matching if no terms found
        if not medical_terms:
            terms = re.findall(rf'\b\w{{{self.min_term_length},}}\b', prompt.lower())
            medical_terms = [
                self.correct_spelling(term) for term in terms
                if self.correct_spelling(term) in MEDICAL_TERMS
            ]   

        return list(set(medical_terms))

    def truncate_prompt(self, prompt: str, max_chars: int = 1000) -> str:
        if not prompt:
            return ""
        return prompt[:max_chars]

    def batch_predict(self, prompts: List[str], batch_size: int = 8, min_score: float = 0.0) -> List[List[str]]:
        if not prompts:
            return []
            
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = [self.predict(prompt, min_score) for prompt in batch]
            results.extend(batch_results)
        return results

# medical_ner_llm = MedicalNERLLM()
# question = 'How does obesity contribute to type 2 diabetes in individuals with a sedentary lifestyle'
# medical_ner_llm.predict(question)