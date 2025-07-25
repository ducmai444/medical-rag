import requests
from langdetect import detect

class UMLS_API:
    def __init__(self, apikey, version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"

    def search_cui(self, query, page_size=10, max_pages=5):
        cui_results = []

        try:
            page = 1
            size = 1
            query = {"string": query, "apiKey": self.apikey, "pageNumber": page, "pageSize": size}
            r = requests.get(self.search_url, params=query)
            r.raise_for_status()
            r.encoding = 'utf-8'
            outputs = r.json()

            items = outputs["result"]["results"]

            if len(items) == 0:
                print("No results found.\n")

            for result in items:
                cui_results.append((result["ui"], result["name"]))

        except Exception as except_error:
            print(except_error)

        return cui_results

    def get_definitions(self, cui):
        try:
            suffix = self.content_suffix.format(cui, "definitions", self.apikey)
            r = requests.get(self.content_url + suffix)
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs = r.json()

            return outputs["result"]
        except Exception as except_error:
            print(except_error)

    @staticmethod
    def remove_duplicate_umls(kg_triples):
        normalized_triples = set()
        result = []
        
        for triple in kg_triples:
            # Normalize by converting to lowercase and handling None values
            normalized = {
                'relatedFromIdName': triple.get('relatedFromIdName', '').lower(),
                'additionalRelationLabel': triple.get('additionalRelationLabel', '').lower(),
                'relatedIdName': triple.get('relatedIdName', '').lower()
            }
            
            # Create tuple for comparison (immutable for set)
            normalized_tuple = (
                normalized['relatedFromIdName'],
                normalized['additionalRelationLabel'],
                normalized['relatedIdName']
            )
            
            # Only add if we haven't seen this normalized form before
            if normalized_tuple not in normalized_triples:
                normalized_triples.add(normalized_tuple)
                result.append(triple)
            
        return result

    def get_relations(self, cui, pages=25, language="ENG"):
        all_relations = []

        try:
            for page in range(1, pages + 1):
                suffix = self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}&sabs=SNOMEDCT_US,MSH,ICD10CM,LNC,RXNORM,CPT,NCI,HL7V2.5"
                r = requests.get(self.content_url + suffix)
                r.raise_for_status()
                r.encoding = "utf-8"
                outputs = r.json()

                page_relations = outputs.get("result", [])
                all_relations.extend(page_relations)

            rels = []
            for rel in all_relations:
                related_from = rel.get("relatedFromIdName", "").strip()
                relation_label = rel.get("additionalRelationLabel", "")
                related_to = rel.get("relatedIdName", "")

                triplet = {
                    "relatedFromIdName": related_from,
                    "additionalRelationLabel": relation_label,
                    "relatedIdName": related_to
                }

                rels.append(triplet)

            
            rels = UMLS_API.remove_duplicate_umls(rels)

            return rels
            
        except Exception as except_error:
            print(except_error)
