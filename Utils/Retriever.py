import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch

# 전역 변수 선언
bi_encoder = None
cause_embeddings = None
environment_embeddings = None
answer_embeddings = None
bm25_cause = None
bm25_env = None
train_data = None  # pandas DataFrame (train_data)

def ready_retriever(model_name, train_df):
    """
    모델과 BM25 인덱스, 그리고 corpus 임베딩을 준비합니다.

    Parameters:
        model_name (str): SentenceTransformer 모델 이름.
        train_df (pd.DataFrame): 사고 원인(accident_cause), 사고 환경(accident_environment),
                                 질문(question), 정답(answer) 컬럼을 가진 DataFrame.
    """
    global bi_encoder, cause_embeddings, environment_embeddings, answer_embeddings, bm25_cause, bm25_env, train_data
    train_data = train_df

    print(f"Loading model: {model_name}")
    bi_encoder = SentenceTransformer(model_name)

    # 사고 원인 임베딩 계산
    print("Computing Cause embeddings for train_data...")
    cause_embeddings = bi_encoder.encode(
        train_data.accident_cause.to_list(),
        convert_to_tensor=True,
        show_progress_bar=True
    )
    print("Cause embeddings computed.")

    # 사고 환경 임베딩 계산
    print("Computing Environment embeddings for train_data...")
    environment_embeddings = bi_encoder.encode(
        train_data.accident_environment.to_list(),
        convert_to_tensor=True,
        show_progress_bar=True
    )
    print("Environment embeddings computed.")

    # 사고 환경 임베딩 계산
    print("Computing Answer embeddings for train_data...")
    answer_embeddings = bi_encoder.encode(
        train_data.answer.to_list(),
        convert_to_tensor=True,
        show_progress_bar=True
    )
    print("Answer embeddings computed.")



    # BM25를 위한 Corpus 준비 (사고 원인과 환경 텍스트)
    print("Computing BM25 corpus...")
    cause_corpus = train_data.accident_cause.to_list()
    env_corpus = train_data.accident_environment.to_list()
    tokenized_cause_corpus = [doc.lower().split() for doc in cause_corpus]
    tokenized_env_corpus = [doc.lower().split() for doc in env_corpus]
    bm25_cause = BM25Okapi(tokenized_cause_corpus)
    bm25_env = BM25Okapi(tokenized_env_corpus)
    print("BM25 corpus for accident_cause and accident_environment prepared.")

def retriever(accident_cause, accident_environment, top_k=5, alpha=0.8, beta=0.2, threshold=0.0):
    """
    주어진 사고 원인과 환경에 대해 BM25 및 LLM 기반 유사도 점수를
    가중치(alpha, beta)를 적용하여 결합한 후 상위 top_k 문서를 반환합니다.

    Parameters:
        accident_cause (str): 사고 원인 텍스트.
        accident_environment (str): 사고 환경 텍스트.
        top_k (int): 반환할 문서 수 (기본값 5).
        alpha (float): LLM 기반 유사도 가중치.
        beta (float): BM25 기반 유사도 가중치.

    Returns:
        str: 상위 문서들의 사례 번호, 질문, 정답을 포함하는 포맷된 문자열.
    """
    global bi_encoder, cause_embeddings, environment_embeddings, bm25_cause, bm25_env, train_data

    # LLM 기반 유사도 계산
    cause_query = bi_encoder.encode(accident_cause, convert_to_tensor=True)
    environment_query = bi_encoder.encode(accident_environment, convert_to_tensor=True)
    cause_similarities = util.cos_sim(cause_query, cause_embeddings)
    environment_similarities = util.cos_sim(environment_query, environment_embeddings)
    # LLM 점수 결합 (alpha 적용)
    combined_llm_scores = cause_similarities * alpha + environment_similarities * alpha

    # BM25 기반 유사도 계산
    query_cause = accident_cause.lower().split()
    query_env = accident_environment.lower().split()
    cause_scores = bm25_cause.get_scores(query_cause)
    env_scores = bm25_env.get_scores(query_env)
    # BM25 점수 결합 (beta 적용)
    combined_bm25_scores = np.array(cause_scores) * beta + np.array(env_scores) * beta
    # 0~1 범위 스케일링 (Min-Max Normalization)
    min_val = np.min(combined_bm25_scores)
    max_val = np.max(combined_bm25_scores)
    combined_bm25_scores = (combined_bm25_scores - min_val) / (max_val - min_val)

    # 최종 점수: BM25 점수에 beta, LLM 점수에 alpha를 각각 곱해 결합
    combined_scores = combined_bm25_scores * beta + combined_llm_scores.cpu().numpy() * alpha

    # 상위 top_k 후보 추출
    indexed_scores = list(enumerate(combined_scores[0].tolist()))
    top_candidates = sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:top_k]

    formatted_docs = ""
    scores = []
    num = 1
    for idx, score in top_candidates:
        if score < threshold:
            break
        scores.append(score)
        question = train_data.loc[idx, 'question']
        answer = train_data.loc[idx, 'answer']
        formatted_docs += f"사례 {num}.\n사건 정보: {question}\n재발방지대책 및 향후조치계획: {answer}\n\n"
        num += 1

    return formatted_docs.strip(), scores


def post_retriever(query_text, top_k=5, threshold=0.0):
    """
    주어진 query_text를 통해 answer_embeddings와 유사도를 계산하고,
    상위 top_k 결과(문서 사례)를 반환합니다.

    Parameters:
        query_text (str): 검색 질의 문장
        top_k (int): 추출할 결과 개수 (기본 5)
        threshold (float): 특정 점수 미만은 제외할 임계값 (기본 0.0)

    Returns:
        (str, list): 포맷된 사례 텍스트와 각 사례별 점수 리스트
    """
    global bi_encoder, answer_embeddings, train_data

    # 쿼리 임베딩 계산
    query_embedding = bi_encoder.encode(query_text, convert_to_tensor=True)

    # answer_embeddings와 코사인 유사도 계산
    answer_similarities = util.cos_sim(query_embedding, answer_embeddings)

    # 상위 top_k 후보 추출
    indexed_scores = list(enumerate(answer_similarities[0].tolist()))
    top_candidates = sorted(indexed_scores, key=lambda x: x[1], reverse=True)[:top_k]

    formatted_docs = ""
    scores = []
    num = 1
    for idx, score in top_candidates:
        if score < threshold:
            break
        scores.append(score)
        # 여기서는 train_data 내부에서 question, answer 등을 참조
        question = train_data.loc[idx, 'question']
        answer = train_data.loc[idx, 'answer']
        formatted_docs += f"예시 {num}.\n사건 정보: {question}\n재발방지대책 및 향후조치계획: {answer}\n\n"
        num += 1

    return formatted_docs.strip(), scores


# retriever 함수를 체인에 사용할 수 있도록 래핑하는 함수
def retriever_runnable(inputs):
    """
    입력 딕셔너리(inputs)는 다음 키들을 포함해야 합니다.
      - "accident_cause": 사고 원인 텍스트
      - "accident_environment": 사고 환경 텍스트
      - "accident_info": 기타 사고 정보 (프롬프트에 그대로 전달)

    반환값은 "retrieved_documents"를 포함하는 딕셔너리입니다.
    """
    accident_cause = inputs.get("accident_cause", "")
    accident_environment = inputs.get("accident_environment", "")
    # retriever() 함수 호출 (여기서는 top_k=5, alpha=0.8, beta=0.2 예시)
    retrieved_docs, scores = retriever(accident_cause, accident_environment, top_k=3, alpha=0.8, beta=0.2, threshold=0.0)
    return retrieved_docs

