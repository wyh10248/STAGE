import pandas as pd

# 读取得分矩阵，行是微生物，列是疾病
score_df = pd.read_csv('../case_study/final_prediction_score.csv', header=None)
score_min = score_df.min().min()
score_max = score_df.max().max()
score_df = (score_df - score_min) / (score_max - score_min)


# 读取微生物和疾病名称
microbe_names = pd.read_csv('../case_study/microbe.csv', header=None)[0].tolist()
disease_names = pd.read_csv('../case_study/disease.csv', header=None)[0].tolist()

# 给矩阵加行列索引（便于之后处理）
score_df.index = microbe_names
score_df.columns = disease_names

# 用于收集结果
result_rows = []

# 对每个疾病（即每列）排序，取前30名
for disease in score_df.columns:
    top30 = score_df[disease].sort_values(ascending=False).head(30)
    for rank, (microbe, score) in enumerate(top30.items(), start=1):
        result_rows.append([microbe, disease, score, rank])

# 转换为DataFrame
result_df = pd.DataFrame(result_rows, columns=["Microbe", "Disease", "Score", "Rank"])

# 保存为Excel
result_df.to_excel('../case_study/top30_microbes_per_disease1.xlsx', index=False)
