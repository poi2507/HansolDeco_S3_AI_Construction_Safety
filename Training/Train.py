
import pandas as pd
from tqdm.auto import tqdm
import gc
import numpy as np

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader
from Evaluate.Matrix import cosine_similarity_sbert, jaccard_similarity, final_score, compute_score


def Training(
    model,
    data_loader,
    tokenizer,
    optimizer,
    device="cuda",
    accumulation_steps=4):
    """
    Causal LM 방식으로 단일 루프(한 번의 데이터셋 순회) 학습을 수행.
    - 에폭(epochs) 반복은 이 함수 밖에서 감싸서 관리.
    - Gradient Accumulation만 적용, Validation / Scheduler 등은 없음.

    Args:
        model: 학습할 PyTorch 모델 (예: AutoModelForCausalLM)
        train_loader: 학습용 DataLoader (input_ids, attention_mask 등 batch 제공)
        tokenizer: 해당 모델의 토크나이저
        optimizer: 옵티마이저 (예: AdamW)
        device: 'cuda' 또는 'cpu'
        accumulation_steps: Gradient Accumulation에 사용할 스텝 수

    Returns:
        float: 단일 순회(에폭) 종료 시 평균 학습 로스
    """
    model.config.use_cache = False
    model.train()

    total_loss = 0.0
    step_count = 0
    global_step = 0

    optimizer.zero_grad()

    # train_loader 한 번 순회
    for i, data in enumerate(tqdm(data_loader, desc="Training"), start=1):
        input_ids = data['input_ids'].to(device, dtype=torch.long)
        attention_mask = data['attention_mask'].to(device, dtype=torch.long)
        labels = input_ids  # Causal LM에서 일반적으로 labels=input_ids

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Gradient Accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # 일정 스텝마다 optimizer.step()
        if (i % accumulation_steps) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # 스텝별 누적 로스 업데이트 (accumulation_steps 곱 적용)
        total_loss += loss.item() * accumulation_steps
        step_count += 1
        global_step += 1

        # 주기적으로 로스 출력
        if (global_step % 100) == 0:
            avg_loss = total_loss / step_count
            print(f"Global Step {global_step}, Avg Train Loss: {avg_loss:.6f}")

    avg_total_loss = total_loss / step_count
    print(f"\nSingle pass training complete. Average Loss: {avg_total_loss:.6f}")
    gc.collect()
    torch.cuda.empty_cache()

    return avg_total_loss

def Evaluating(model, embedding_model, tokenizer, validation_data, device="cuda"):
    model.config.use_cache = True
    model.eval()
    test_results = []

    for i in tqdm(range(len(validation_data)), desc="Validation"):
        question = validation_data.loc[i, 'question']
        context = f'''아래는 건설 현장에서 발생한 단일 사건 정보입니다. 이 정보를 토대로, 불필요한 부연 설명을 줄이고 핵심적인 재발 방지 대책 및 향후 조치 계획을 간결하게 작성하세요. 답변은 한국어로 작성하며, 문장 말미에 불필요한 기호나 종결 어구를 사용하지 말고, 하나의 단락으로 자연스럽게 연결된 문장으로 완성하세요.\n\n주어진 사건 정보: {question}\n재발방지대책 및 향후조치계획:'''
        batch = tokenizer(context, return_tensors='pt').to(device)
        input_len = batch["input_ids"].shape[1]
        output_ids = model.generate(**batch)
        decoded_text = tokenizer.decode(output_ids[0][input_len:]  ,skip_special_tokens=True)
        test_results.append(decoded_text)

    final_score = compute_score(test_results, validation_data['answer'].tolist(), embedding_model)

    return final_score, test_results
