import pandas as pd
import numpy as np

# 读取数据文件
file_path = 'MDA.txt'
df = pd.read_csv(file_path, sep='\t')

# 获取所有唯一的疾病和微生物名称
diseases = df['Disease'].unique()
microbes = df['Microbe'].unique()

# 创建一个疾病与微生物的空矩阵，行是微生物，列是疾病
matrix = np.zeros((len(microbes), len(diseases)), dtype=int)

# 创建疾病和微生物的映射，用于矩阵的索引
microbe_index = {microbe: idx for idx, microbe in enumerate(microbes)}
disease_index = {disease: idx for idx, disease in enumerate(diseases)}

# 创建一个字典来存储每个微生物对应的PMID
microbe_pmid = {}

# 填充矩阵并收集PMID
for _, row in df.iterrows():
    disease = row['Disease']
    microbe = row['Microbe']
    pmid = row['PMID']
    
    # 将对应的矩阵位置设置为1，表示该微生物与疾病相关
    matrix[microbe_index[microbe], disease_index[disease]] = 1
    
    # 如果该微生物已经存在，追加PMID
    if microbe not in microbe_pmid:
        microbe_pmid[microbe] = set()
    microbe_pmid[microbe].add(pmid)

# 将PMID合并为一个字符串，方便后续保存
for microbe in microbe_pmid:
    microbe_pmid[microbe] = '; '.join(map(str, microbe_pmid[microbe]))

# 将结果转换为 DataFrame，更易于查看
matrix_df = pd.DataFrame(matrix, index=microbes, columns=diseases)

# 将PMID列添加到 DataFrame
matrix_df['PMID'] = matrix_df.index.map(microbe_pmid)

# 保存邻接矩阵和PMID到 CSV 文件
output_file = 'adjacency_matrix.csv'
matrix_df.to_csv(output_file)

print(f"邻接矩阵和PMID已保存到 {output_file}")
