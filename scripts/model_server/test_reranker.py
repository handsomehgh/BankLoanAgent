# author hgh
# version 1.0
# author hgh
# version 1.0
import requests
import json

url = "http://127.0.0.1:8080/rerank"

data = {
    "query": "房贷利率现在是多少？",
    "documents": [
        "一、个人住房贷款 > 利率政策 | 首套房贷款5年以上LPR为4.2%，加30BP后为4.5%。",
        "四、合同签订 > 4.3 费用确认 | 客户需逐项看清费用明细，拒绝任何合同外收费。",
        "二、还款方式 > 等额本息 | 每月还款金额固定，便于预算。",
        "一、个人住房贷款 > 利率政策 | 二套房利率不低于LPR+60BP，当前约为4.95%。"
    ]
}

try:
    response = requests.post(url, json=data, timeout=30)
    response.raise_for_status()
    result = response.json()
    print("请求成功！")
    scores = result["scores"]
    sorted_indices = result["sorted_indices"]

    print(f"共返回 {len(scores)} 个文档的分数：")
    for rank, idx in enumerate(sorted_indices):
        print(f"  排名 {rank+1}: 文档 {idx} (分数: {scores[idx]:.4f})")
        doc_preview = data["documents"][idx][:60] + "..." if len(data["documents"][idx]) > 60 else data["documents"][idx]
        print(f"    内容: {doc_preview}")

except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
except json.JSONDecodeError:
    print("响应不是有效的 JSON")
    print(response.text)