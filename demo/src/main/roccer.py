from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import RocCurveDisplay
from math import ceil
import matplotlib.pyplot as plt


# scores_d = {}
# scores_d["Abuse/028"] = [0,0,0,0,0.8,0,0,0,0,0,0,0,0,0,0.7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

normal_out = ["Normal/059", "Normal/129","Normal/189","Normal/478","Normal/656","Normal/877","Normal/887","Normal/898","Normal/901","Normal/925",]

def by_score(obj):
    return obj[0]

def compute_score_seq(scores_d, antn):
    anno_seq = []
    score_seq = []
    normal_maxis = []
    for lbl, scores in scores_d.items(): #TODO note this is extremely dependent on keys in dict
        scores = [float(i) for i in scores]
        cat, num = lbl.split('/')

        n_scores = 32
        anno_profile = np.zeros(n_scores)
        split_indices = []
        
        # ANOMALOUS VIDEOS
        if not "Normal" in cat:
            # GET ANNOTATIONS
            vid_anno = antn[antn[:, 0] == cat]
            vid_anno = np.squeeze(vid_anno[vid_anno[:, 1] == num])
            # vid_anno = ["Abuse", "028", "1412", "165"," 240", "600", "800"]  
            
            if len(vid_anno) == 0:
                pass
                print("\t\t************************WARNING: annotation not found for {}/{}".format(cat, num))
            else:
                n_frames = int(vid_anno[2])
                for pt in range(3, len(vid_anno), 2):
                    if vid_anno[pt] != "-1":
                        start_anom = int(vid_anno[pt])
                        end_anom = int(vid_anno[pt+1])
                        start_anom = max(0, int((start_anom/n_frames)*n_scores - 4))
                        end_anom = min(n_scores, ceil((end_anom/n_frames)*n_scores + 4))
                        for i in range(start_anom, end_anom):
                            anno_profile[i] = 1
                            
                        split_indices.append(max(0,start_anom-1))
                        split_indices.append(end_anom)

            # print("START")            
            # print("scores")
            # print(scores)
            # print("anno")
            # print(anno_profile)
            # print("split indices")
            # print(split_indices)
            
            # print("split scores")
            split_scores = np.array_split(scores, split_indices)
            # print(split_scores)
            
            # print("split annos")
            split_annos = np.array_split(anno_profile, split_indices)
            # print(split_annos)            


            for sc in split_scores:
                if len(sc.tolist()) != 0:
                    score_seq.append(max(sc))
            for sc in split_annos:
                if len(sc.tolist()) != 0:
                    anno_seq.append(max(sc))
                    
        else:
            if not lbl in normal_out:
                # normal_maxis.append((max(scores), lbl))
                score_seq.append(max(scores))
                anno_seq.append(0)
            
    # for tup in sorted(normal_maxis, key=by_score):
        # print(tup)
            
    # print(anno_seq)
    # print(score_seq)
    return anno_seq, score_seq
    
    

    
def compute_ROC(anno_seq, score_seq):
    RocCurveDisplay.from_predictions(anno_seq, score_seq)
    plt.show()
    
    

# anno_seq, score_seq = compute_score_seq(scores_d, None)
# ROC = compute_ROC(anno_seq, score_seq)