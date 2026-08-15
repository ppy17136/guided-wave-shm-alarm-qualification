from __future__ import annotations
import numpy as np

def false_alarms_per_month(scores, labels, month_ids, threshold):
    scores,labels,month_ids=map(np.asarray,(scores,labels,month_ids)); alarm=(scores>=threshold)&(labels==0); months=np.unique(month_ids[labels==0]); return float(sum(alarm[month_ids==m].sum() for m in months)/len(months)) if len(months) else float("nan")

def block_conformal_threshold(calibration_scores, alpha=0.01):
    values=np.sort(np.asarray(calibration_scores,float))
    if not values.size: raise ValueError("Empty calibration scores")
    rank=int(np.ceil((values.size+1)*(1-alpha)))-1
    return float(values[min(max(rank,0),values.size-1)])

