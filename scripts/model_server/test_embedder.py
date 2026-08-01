# author hgh
# version 1.0
import requests
import json
import numpy as np

# 服务地址（请根据实际部署修改端口）
url = "http://127.0.0.1:8001/v1/embeddings"

# 构造测试文本（支持批量，可以发送多条）
texts = [
    "提前还款的话，违约金怎么算？",
    "公积金贷款利率会不会变动？",
    "贷款审批需要多长时间？"
]

# 请求数据格式
data = {"input": texts}

try:
    # 发送 POST 请求
    response = requests.post(url, json=data, timeout=30)
    response.raise_for_status()
    result = response.json()
    print(result)

    embeddings = result["data"]
    print(f"请求成功！共返回 {len(embeddings)} 条嵌入向量。")

    if embeddings:
        # 转换为 numpy 数组，便于查看形状
        emb_array = np.array(embeddings)
        print(f"嵌入矩阵形状: {emb_array.shape}")       # 应为 (文本条数, 768)
        print(f"第一条嵌入的前5个值: {emb_array[0][:5]}")
        print(f"嵌入向量L2范数（应接近1）: {np.linalg.norm(emb_array[0]):.4f}")

except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
except json.JSONDecodeError:
    print("响应不是有效的 JSON")
    print(response.text)
