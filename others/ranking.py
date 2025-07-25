import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from umlsbert import UMLSBERT

umlsbert = UMLSBERT()

# PPR Ranking
def ppr_ranking(query, relations, main_entity, top_k=150):
    # Encode query and triplets using UmlsBERT
    relation_texts = [f"{rel.get('relatedFromIdName', '')} {rel.get('additionalRelationLabel', '').replace('_', ' ')} {rel.get('relatedIdName', '')}" for rel in relations]
    embeddings = umlsbert.batch_encode([query] + relation_texts)
    query_embedding = embeddings[0]
    relation_embeddings = embeddings[1:]
    
    # Calculate cosine similarity to initialize weights
    cos_sims = cosine_similarity([query_embedding], relation_embeddings)[0]
    weights = {rel.get("relatedIdName", ""): sim for rel, sim in zip(relations, cos_sims)}
    
    # Build subgraph
    G = nx.DiGraph()
    for rel in relations:
        e_i = rel.get("relatedFromIdName", "")
        e_j = rel.get("relatedIdName", "")
        r = rel.get("additionalRelationLabel", "").replace("_", " ")
        G.add_edge(e_i, e_j, relation=r)
    
    # Run Personalized PageRank
    personalization = {e: 1.0 if e == main_entity else weights.get(e, 0.1) for e in G.nodes}
    pr = nx.pagerank(G, alpha=0.85, personalization=personalization, max_iter=100)
    
    # Rank triplets based on PPR score of e_j
    ranked_rels = sorted(
        [(pr.get(rel.get("relatedIdName", ""), 0), rel) for rel in relations],
        key=lambda x: x[0],  # Sort by PPR score
        reverse=True
    )[:top_k]

    return [rel[1] for rel in ranked_rels]

# MMR Ranking
def get_similarity(query_emb, rel_emb):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(query_emb, rel_emb)

def most_similar_relation(query_embedding, relation_embeddings):
    similarities = [get_similarity([query_embedding], [rel_emb]) for rel_emb in relation_embeddings]
    index = similarities.index(max(similarities))
    return index

def calculate_rerank_scores(query_embedding, relation_embeddings, rerank_relations_indices, base_weight=0.1, delta_weight=0.01):
    scores = []
    for i, rel_emb in enumerate(relation_embeddings):
        if i not in rerank_relations_indices:
            query_similarity = get_similarity([query_embedding], [rel_emb])
            
            if rerank_relations_indices:
                rel_similarities = [get_similarity([relation_embeddings[j]], [rel_emb]) for j in rerank_relations_indices]
                avg_rel_similarity = sum(rel_similarities) / len(rel_similarities)
            else:
                avg_rel_similarity = 0
            
            weight_factor = base_weight + delta_weight * len(rerank_relations_indices)
            
            score = query_similarity - weight_factor * avg_rel_similarity
            scores.append((i, score))
    return scores

def MMR_reranking(query, relations, top_k=10):
    rels = []
    relation_texts = [query] + [f"{rel.get('relatedFromIdName', '')} {rel.get('additionalRelationLabel', '').replace('_', ' ')} {rel.get('relatedIdName', '')}" for rel in relations]
    embeddings = umlsbert.batch_encode(relation_texts)
    query_embedding = embeddings[0]
    relation_embeddings = embeddings[1:]

    rerank_relations_indices = [most_similar_relation(query_embedding, relation_embeddings)]

    while len(rerank_relations_indices) < 20 and len(relation_embeddings) > len(rerank_relations_indices):
        rerank_scores = calculate_rerank_scores(query_embedding, relation_embeddings, rerank_relations_indices)
        rerank_scores.sort(key=lambda x: x[1], reverse=True)
        for score in rerank_scores:
            if score[0] not in rerank_relations_indices:
                rerank_relations_indices.append(score[0])
                break
    
    rerank_relations = [relations[i] for i in rerank_relations_indices]
    rerank_relations = rerank_relations[:top_k]

    for rel in rerank_relations:
        related_from_id_name = rel.get("relatedFromIdName")
        additional_relation_label = rel.get("additionalRelationLabel").replace("_", " ")
        related_id_name = rel.get("relatedIdName")

        rel = {
            "relatedFromIdName": related_from_id_name,
            "additionalRelationLabel": additional_relation_label,
            "relatedIdName": related_id_name
        }
        rels.append(rel)

    return rels

def similarity_score(query, relations, top_k=10):
    def get_similarity(query_emb, rel_emb):
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity([query_emb], [rel_emb])[0][0]
    
    relation_texts = [query] + [f"{rel.get('relatedFromIdName', '')} {rel.get('additionalRelationLabel', '').replace('_', ' ')} {rel.get('relatedIdName', '')}" for rel in relations]
    embeddings = umlsbert.batch_encode(relation_texts)
    query_embedding = embeddings[0]
    relation_embeddings = embeddings[1:]
    
    relation_scores = [(get_similarity(query_embedding, rel_emb), rel) for rel_emb, rel in zip(relation_embeddings, relations)]
    relation_scores.sort(key=lambda x: x[0], reverse=True)
    rank_rels = relation_scores[:top_k]

    rels = []
    for _, rel in rank_rels:
        rel = {
            "relatedFromIdName": rel.get("relatedFromIdName", ""),
            "additionalRelationLabel": rel.get("additionalRelationLabel", "").replace("_", " "),
            "relatedIdName": rel.get("relatedIdName", "")
        }
        rels.append(rel)

    return rels

