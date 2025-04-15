
import pandas as pd
import glob
import os
import re
import pdfplumber
from tqdm.auto import tqdm


def data_preprocessing(train, test):

    # 데이터 전처리
    train['공사종류(대분류)'] = train['공사종류'].str.split(' / ').str[0]
    train['공사종류(중분류)'] = train['공사종류'].str.split(' / ').str[1]
    train['공종(대분류)'] = train['공종'].str.split(' > ').str[0]
    train['공종(중분류)'] = train['공종'].str.split(' > ').str[1]
    train['사고객체(대분류)'] = train['사고객체'].str.split(' > ').str[0]
    train['사고객체(중분류)'] = train['사고객체'].str.split(' > ').str[1]

    test['공사종류(대분류)'] = test['공사종류'].str.split(' / ').str[0]
    test['공사종류(중분류)'] = test['공사종류'].str.split(' / ').str[1]
    test['공종(대분류)'] = test['공종'].str.split(' > ').str[0]
    test['공종(중분류)'] = test['공종'].str.split(' > ').str[1]
    test['사고객체(대분류)'] = test['사고객체'].str.split(' > ').str[0]
    test['사고객체(중분류)'] = test['사고객체'].str.split(' > ').str[1]

    # 훈련 데이터 통합 생성
    combined_training_data = train.apply(
        lambda row: {
            "question": (
                f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
                f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
                f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
                f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
                f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
            ),
            "answer": row["재발방지대책 및 향후조치계획"],

            'accident_cause': f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다.",
            'accident_environment': f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')'"

        },
        axis=1
    )

    # DataFrame으로 변환
    combined_training_data = pd.DataFrame(list(combined_training_data))

    # 테스트 데이터 통합 생성
    combined_test_data = test.apply(
        lambda row: {
            "question": (
                f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
                f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
                f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
                f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
                f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
            ),
            'accident_cause': f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다.",
            'accident_environment': f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')'"

        },
        axis=1
    )

    # DataFrame으로 변환
    combined_test_data = pd.DataFrame(list(combined_test_data))

    return combined_training_data, combined_test_data

def insert_word_bridging_rows(
    df: pd.DataFrame,
    text_col: str = 'text',
    bridging_end: int = 64,
    bridging_start: int = 64
) -> pd.DataFrame:
    """
    df: 원본 DataFrame (예: index, text 열이 존재)
    text_col: 텍스트가 들어있는 컬럼 이름
    bridging_end: 현재 행의 끝에서 가져올 '단어' 개수
    bridging_start: 다음 행의 앞에서 가져올 '단어' 개수

    반환:
      - 원본 행 + (행 사이) '브릿지' 행이 번갈아 들어간 확장된 DataFrame
        (열: [original_index, bridge, text_col])
    """

    new_rows = []
    n = len(df)

    for i in range(n):
        # (1) 원본 행 추가 (bridge=False)
        original_row = df.iloc[i].to_dict()
        new_rows.append({
            "original_index": i,
            "bridge": False,
            text_col: original_row[text_col]
        })

        # (2) i와 i+1 사이 브리지 행 추가 (단 i+1이 있을 때만)
        if i < n - 1:
            current_text = str(df.iloc[i][text_col])
            next_text = str(df.iloc[i+1][text_col])

            # 단어 기준으로 split
            current_words = current_text.split()
            next_words = next_text.split()

            # 현재 행의 마지막 bridging_end개 단어
            tail_part = current_words[-bridging_end:] if len(current_words) >= bridging_end else current_words
            # 다음 행의 첫 bridging_start개 단어
            head_part = next_words[:bridging_start] if len(next_words) >= bridging_start else next_words

            # 브리지 텍스트 합치기
            bridging_text = " ".join(tail_part) + "\n" + " ".join(head_part)

            new_rows.append({
                "original_index": f"{i}->{i+1}",
                "bridge": True,
                text_col: bridging_text
            })

    df_bridged = pd.DataFrame(new_rows)
    return df_bridged[['text']]

def pdf_to_text(pdf_directory='./Data/건설안전지침', output_file="./Data/all_construction_guide.csv"):

    # 만약 output_file이 이미 존재하면, 그 파일을 불러옴
    if os.path.exists(output_file):
        print(f"'{output_file}' 파일이 이미 존재하여, 이를 불러옵니다.")
        df_existing = pd.read_csv(output_file, encoding='utf-8')
        return df_existing

    # output_file이 없으면, PDF를 읽어 텍스트를 추출
    pdf_files = glob.glob(os.path.join(pdf_directory, '*.pdf'))
    documents = []

    # 예: "KOSHA GUIDE\nC - 08 - 2015\n" 등 제거할 패턴
    #pattern = r"KOSHA GUIDE\nC\s*-\s*\d+\s*-\s*\d+\n"
    #pattern1 = r"(?i)KOSHA GUIDE\nC\s*-\s*\d+\s*-\s*\d+\n"
    patterns = [
    r"(?i)kosha\s*guide\s*c\s*-\s*\d+\s*-\s*\d+",
    r'\([^)]*\)',    # ( )
    r'\[[^\]]*\]',   # [ ]
    r'\{[^}]*\}',     # { }
    r'-\s*\d+\s*-'
    ]


    print("PDF 파일로부터 텍스트를 추출 중입니다...")
    for pdf_file in tqdm(pdf_files):
        with pdfplumber.open(pdf_file) as pdf:
            # 예: PDF 2페이지 이후부터 추출 (상황에 따라 수정)
            for i in range(3, len(pdf.pages)):
                page = pdf.pages[i]
                text = page.extract_text() or ""

                for p in patterns:
                    text = re.sub(p, '', text)
                
                # 추가 전처리(공백 정리 등) 필요 시 여기서 처리
                documents.append(text)

    # 모든 PDF 텍스트 합침
    df = pd.DataFrame(data=documents, columns=['text'])
    df = insert_word_bridging_rows(df)
    # 로컬에 저장
    df = df.dropna().reset_index(drop=True)
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"문서가 '{output_file}' 파일로 생성 및 저장되었습니다.")
    return df
