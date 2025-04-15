from sentence_transformers import SentenceTransformer
import numpy as np

def cosine_similarity_sbert(ref_text, pred_text, embedding_model):
    """
    ref_text, pred_text: 문자열
    S-BERT 임베딩 후 코사인 유사도를 계산하여 반환
    """
    embeddings = embedding_model.encode([ref_text, pred_text], convert_to_numpy=True)
    v_gt = embeddings[0]
    v_pred = embeddings[1]

    # 코사인 유사도 계산
    dot = np.dot(v_gt, v_pred)
    norm = (np.linalg.norm(v_gt) * np.linalg.norm(v_pred))
    if norm == 0.0:
        return 0.0
    return dot / norm

##############################
# 2. Jaccard Similarity 계산 함수
##############################
def jaccard_similarity(ref_text, pred_text):
    """
    ref_text, pred_text: 문자열
    공백 기준 토큰화 후 집합을 구성하여 자카드 유사도 반환
    """
    ref_tokens = set(ref_text.split())
    pred_tokens = set(pred_text.split())

    inter = ref_tokens.intersection(pred_tokens)
    union = ref_tokens.union(pred_tokens)
    if len(union) == 0:
        return 0.0
    return len(inter) / len(union)

##############################
# 2. 스코어 계산 함수
##############################
def final_score(ref_text, pred_text, embedding_model):
    cos_sim = cosine_similarity_sbert(ref_text, pred_text, embedding_model)
    jac_sim = jaccard_similarity(ref_text, pred_text)
    score = 0.7 * max(cos_sim, 0) + 0.3 * max(jac_sim, 0)
    return score

##############################
# 5. 최종 스코어 계산 함수
##############################
def compute_score(references, predictions, embedding_model):
    """
    references, predictions: 문자열 리스트 (길이가 같아야 함)

    1) 각 쌍에 대해 S-BERT 코사인 유사도(CosineSim)와 Jaccard 유사도(JaccardSim)를 구함
    2) 0.7 * max(CosineSim, 0) + 0.3 * max(JaccardSim, 0)를 합산
    3) N으로 나눈 평균 점수를 반환
    """
    n_samples = len(references)
    total_score = 0.0
    total_cos_sim = 0.0
    total_jac_sim = 0.0
    for ref_text, pred_text in zip(references, predictions):
        cos_sim = cosine_similarity_sbert(ref_text, pred_text, embedding_model)
        jac_sim = jaccard_similarity(ref_text, pred_text)

        # 부분 점수 = 0.7 * max(CosineSim, 0) + 0.3 * max(JaccardSim, 0)
        partial_score = 0.7 * max(cos_sim, 0) + 0.3 * max(jac_sim, 0)
        total_score += partial_score
        total_cos_sim += cos_sim
        total_jac_sim += jac_sim


    # 최종 점수
    return {'Final Score' : total_score / n_samples, 'Cosine Score' : total_cos_sim / n_samples, 'Jaccard Score' : total_jac_sim / n_samples}
