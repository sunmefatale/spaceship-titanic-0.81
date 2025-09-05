from pathlib import Path
import os, sys
import pandas as pd

HERE = Path(__file__).resolve()         # 지금 실행 중인 .py의 절대경로
BASE = HERE.parent                      # 그 .py 파일이 있는 폴더
print("▶ THIS FILE:", HERE)            #파일 전체 경로
print("▶ LOOKING IN:", BASE)           #csv를 찾을 기준폴더
print("▶ CWD (현재 작업폴더):", Path.cwd()) #터미널 현재 폴더
print("▶ CSVs in BASE:", [p.name for p in BASE.glob("*.csv")])
print("▶ train.csv exists?", (BASE / "train.csv").exists())

# 경로가 맞으면 읽기
train = pd.read_csv(BASE / "train.csv")
test  = pd.read_csv(BASE / "test.csv")

print(train.shape)
print(train.head())

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("\n=== 기본 정보 ===")
print(train.info())

print("\n=== 결측치 개수(내림차순) ===")
missing = train.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0])

print("\n=== 타깃 분포 (Transported) ===")
print(train["Transported"].value_counts())
print("비율(=True 평균):", train["Transported"].mean())

# 간단 시각화
plt.figure()
sns.countplot(x="Transported", data=train)
plt.title("Target Distribution (Transported)")
plt.show()
