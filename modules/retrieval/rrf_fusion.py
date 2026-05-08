# author hgh
# version 1.0
from collections import defaultdict
from typing import List, Dict

from config.global_constant.fields import CommonFields


def rrf_fusion(result_list: List[List[Dict]],k: int = 60) -> List[Dict]:
    """
    enter a list of multiple results,with each result sorted in descending order by score,
    return the merged list ,sorted by rrf score in descending order,with duplicates removed(based on id)
    """
    score_dict = defaultdict(float)
    id_to_item = {}
    for results in result_list:
        for rank,item in enumerate(results):
            doc_id = item.get(CommonFields.ID)
            if not doc_id:
                continue
            score_dict[doc_id] += 1.0 / (k + rank + 1)
            if doc_id not in id_to_item:
                id_to_item[doc_id] = item
    sorted_ids = sorted(score_dict.keys(),key=lambda x: score_dict[x],reverse=True)
    fused = []
    for doc_id in sorted_ids:
        item = id_to_item[doc_id]
        item["rrf_score"] = score_dict[doc_id]
        fused.append(item)
    return fused


