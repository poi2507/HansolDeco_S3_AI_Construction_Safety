
import pandas as pd
from tqdm.auto import tqdm
import gc
import numpy as np

from torch.utils.data import Dataset, DataLoader, ConcatDataset

def stratified_2d_sampling(df, x_col='x', y_col='y',
                           nx_bins=150, ny_bins=150,
                           total_samples=1000,
                           random_state=None):
    """
    1) (x_col, y_col) 2차원 공간을 nx_bins x ny_bins 로 나눈다.
    2) 각 bin에 속한 데이터 중 일정 개수를 무작위로 뽑는다.
       - bin이 너무 빈약하면(해당 bin에 row가 적으면), 그만큼만 뽑는다.
    3) 반환: 2D로 골고루 샘플링된 DataFrame
    --------------------------------------------------------------------------
    - df: 원본 DataFrame (x_col, y_col이 존재)
    - nx_bins, ny_bins: x축, y축을 몇 개 구간으로 나눌지
    - total_samples: 최종적으로 뽑으려는 총 샘플 수
    - random_state: 난수 시드 (재현성)
    """

    # 1) x, y의 최소/최대
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()

    # 2) x, y 구간 (bin edges)
    x_edges = np.linspace(x_min, x_max, nx_bins + 1)
    y_edges = np.linspace(y_min, y_max, ny_bins + 1)

    # 3) 각 행이 어느 bin에 속하는지 (0 ~ nx_bins-1, 0 ~ ny_bins-1)
    x_bin_indices = np.digitize(df[x_col], x_edges) - 1
    y_bin_indices = np.digitize(df[y_col], y_edges) - 1

    x_bin_indices = np.clip(x_bin_indices, 0, nx_bins - 1)
    y_bin_indices = np.clip(y_bin_indices, 0, ny_bins - 1)

    # (중요) 원본 df 복사 후, bin 인덱스 열 추가
    df = df.copy()
    df['_x_bin'] = x_bin_indices
    df['_y_bin'] = y_bin_indices

    # 4) bin별 데이터 개수 확인
    grouped = df.groupby(['_x_bin', '_y_bin'], group_keys=False)

    # 5) bin별 샘플링 개수 계산
    #    전체 bin = nx_bins * ny_bins
    #    각 bin에서 뽑을 목표치 = total_samples / (nx_bins * ny_bins)
    bin_count = nx_bins * ny_bins
    samples_per_bin_float = total_samples / float(bin_count)
    samples_per_bin = int(np.ceil(samples_per_bin_float))  # 반올림하여 int

    # 6) 각 bin에서 sample
    results = []
    for (xb, yb), sub_df in grouped:
        # 해당 bin의 row가 'samples_per_bin'보다 적으면 그만큼만 뽑음
        n_to_sample = min(samples_per_bin, len(sub_df))
        if n_to_sample > 0:
            # 무작위 샘플
            sampled_sub = sub_df.sample(n=n_to_sample, random_state=random_state)
            results.append(sampled_sub)

    df_sampled = pd.concat(results).drop(columns=['_x_bin', '_y_bin'])

    # 만약 뽑힌 결과가 total_samples보다 클 수 있음(ceil했으므로).
    # 혹은 어떤 bin이 비어있으면 적을 수 있음.
    # 여기서는 total_samples에 근접한 양을 뽑게 되므로, 필요하다면 추가 후처리.
    # ex) 과하게 뽑혔다면 df_sampled = df_sampled.sample(n=total_samples, random_state=...)
    # 적게 뽑혔다면 어쩔 수 없음(빈 bin이 많거나 sparse한 경우).

    # 만약 정확히 1000개만 유지하고 싶으면:
    if len(df_sampled) > total_samples:
        df_sampled = df_sampled.sample(n=total_samples, random_state=random_state)

    return df_sampled


class PDFDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve a single row from the DataFrame
        row = self.data.iloc[idx]
        text = row["text"]

        # 5) Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt'
        )
        # Remove batch dimension (size=1)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

class SampleDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token  # EOS 토큰 설정

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve a single row from the DataFrame
        row = self.data.iloc[idx]
        question = row["question"]
        answer = row["answer"]
        context = f'''아래는 건설 현장에서 발생한 단일 사건 정보입니다. 이 정보를 토대로, 불필요한 부연 설명을 줄이고 핵심적인 재발 방지 대책 및 향후 조치 계획을 간결하게 작성하세요. 답변은 한국어로 작성하며, 문장 말미에 불필요한 기호나 종결 어구를 사용하지 말고, 하나의 단락으로 자연스럽게 연결된 문장으로 완성하세요.\n\n주어진 사건 정보: {question}\n재발방지대책 및 향후조치계획: {answer} {self.eos_token}'''

        # 5) Tokenize
        inputs = self.tokenizer(
            context,
            return_tensors='pt'
        )
        # Remove batch dimension (size=1)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs
