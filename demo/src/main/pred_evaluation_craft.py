import time
import numpy as np
import os
import sys
from functools import reduce
from random import shuffle
import copy
import time
from itertools import islice
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import pandas as pd
import streamlit as st
import base64
from itertools import islice
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


root=os.getcwd()
VIDEO_PATH="/home/tomsch/Honours/Honours_Project/test-video-set/all-videos/"
ANNO_FILE=os.path.join(root, "demo", "annotations", "Temporal_Anomaly_Annotation.txt")
min_thresh = 0.1
max_thresh = 0.9
CLASSES = ["A", "N"]

def read_anno_file(f):
    # anno = np.genfromtxt(f, dtype={'names': ('class', 'video', 's1', 'e1', 's2', 'e2'),
    #    'formats': (dtype=str, np.int, np.int, np.int, np.int, np.int)})\
    anno = np.genfromtxt(f, dtype=str, comments="{% comment %}")
    # fw = open(os.path.join(root, "annotations", "video_refs.txt"), "w")
    # for i in range(anno.shape[0]):
        # a = anno[i]
        # tplt = "gdrive/MyDrive/honours_project/c3d_features/data/anomaly-videos/{}/{}{}_x264.mp4".format(a[0], a[0], a[1])
        # fw.write("{}\n".format(tplt))
    # fw.close()
    return anno


def verify_anom_fc6(scores, n_frame, anom_window, threshold, buffer, prev_res):
    #TODO -200 for annMIL with caffe c3d
    
    n_frm = n_frame
    n_scores = len(scores)
    start_anom = int(((anom_window[0]-buffer)/n_frm)*n_scores)
    end_anom = int((min(anom_window[1]+buffer, n_frm)/n_frm)*n_scores)
    anno_sgms = sorted(list(set([max(0,start_anom-1), end_anom-1])))
    # print("n_frm:{}".format(n_frm))
    # print("n_scores:{}".format(n_scores))
    # print("start_anom_frm:{}".format(anom_window[0]))
    # print("end_anom_frm:{}".format(anom_window[1]))
    # print("start_anom_scores:{}".format(start_anom))
    # print("end_anom_scores:{}".format(end_anom))
    

    for i in range(anno_sgms[0], anno_sgms[1]):
        if scores[i] > threshold:
            
            return True, anno_sgms
    return (False or prev_res), anno_sgms
        

def frame_averaging(dataset):
    sims_d_div16 = {}
    for lbl, data in dataset.items():
        avgd_data  = []
        sz  = len(data)/16
        for i in range(1, int(sz)):
            if i == sz-1:
                avgd_data.append(np.average(data[(i-1)*16:]))
            else:
                avgd_data.append(np.average(data[(i-1)*16: i*16]))
        sims_d_div16[lbl] = avgd_data
    return sims_d_div16      

def individual_scale(sims, cutoff):
    scaled_sims = {}
    for lbl, sim in sims.items():
        s = np.std(sim)
        u = np.average(sim)
            
        sim_sc = [(x-u)/s for x in sim]
        sim_sc = np.clip(np.asarray(sim_sc), cutoff, np.max(sim_sc)) #atleast clip anom scores that are less than 1 std devs from avg of batch
        # mx = max(sim_sc)
        # mn = min(sim_sc)
        # sim_sc_norm = [(x-mn)/(mx-mn) for x in sim_sc]
        scaled_sims[lbl] = sim_sc
    return scaled_sims
      

def relative_scale(sims, group, cutoff):
    scaled_sims = {}
    batch_values = []
    count = 0
    lbls = list(sims.keys())
    shuffle(lbls)
    
    for lbl in lbls:
        sim = sims[lbl]
        batch_values.extend(sim)
        count +=1
        if count % group == 0 or count == len(sims):
            if count % group == 0:
                subset = islice(sims.items(), count-group, count)
            else:
                back_count = 0
                while (int(count-back_count) % group != 0):
                    back_count+=1
                subset = islice(sims.items(), count-back_count, count)
                
                
            s = np.std(batch_values)
            u = np.average(batch_values)
            m = np.max(batch_values)
            ms = (m-u)/s
            
            for lb, si in subset:
                sim_sc = [(x-u)/s for x in si]
            # print("initial sim len: ", len(sim))
                # mms = MinMaxScaler()
                # scaled = mms.fit_transform(clipped.reshape(1,-1))
                sim_sc = np.clip(np.asarray(sim_sc), cutoff, np.max(sim_sc)) #atleast clip anom scores that are less than 1 std devs from avg of batch
                # mx = max(sim_sc)
                # mn = min(sim_sc)
                # sim_sc_norm = [(x-mn)/(mx-mn) for x in sim_sc]
                scaled_sims[lb] = sim_sc
                
            batch_values = []
            
    return scaled_sims
            
def rangeto1(scores_d):
    scaled_d = {}
    for lbl, scores in scores_d.items():
        mx = max(scores)
        mn = min(scores)
        sim_sc_norm = [(x-mn)/(mx-mn) for x in scores]
        scaled_d[lbl] = sim_sc_norm
    return scaled_d

def bulkrangeto1(sims, group, cutoff):
    scaled_sims = {}
    batch_values = []
    count = 0
    lbls = list(sims.keys())
    shuffle(lbls)
    
    for lbl in lbls:
        sim = sims[lbl]
        batch_values.extend(sim)
        count +=1
        if count % group == 0 or count == len(sims):
            if count % group == 0:
                subset = islice(sims.items(), count-group, count)
            else:
                back_count = 0
                while (int(count-back_count) % group != 0):
                    back_count+=1
                subset = islice(sims.items(), count-back_count, count)
                
                
            mx = np.max(batch_values)
            mn = np.min(batch_values)
            
            for lb, si in subset:
            # print("initial sim len: ", len(sim))
                # mms = MinMaxScaler()
                # scaled = mms.fit_transform(clipped.reshape(1,-1))
                sim_sc = [(x-mn)/(mx-mn) for x in si]
                sim_sc = np.clip(np.asarray(sim_sc), 0, 1) #atleast clip anom scores that are less than 1 std devs from avg of batch
                scaled_sims[lb] = sim_sc
                
            batch_values = []
            
    return scaled_sims
    
def peak_window(scores_d, wdw):
    
    transform_scores_d = {}
    print("[PEAK WDW] window size: {}".format(wdw))
    for lbl, scores in scores_d.items():
        transform_scores = []
        for f in range(0, len(scores)):
            # print("[PEAK WDW] frame: ", f)
            if f > wdw and (len(scores)-f-1) > wdw:

                before = sum(scores[f-wdw:f])
                after = sum(scores[f:f+wdw])
                step = after - before
                # print("\t[PEAK WDW] valid f -> score is ", scores[f])
                # print("\t[PEAK WDW] valid f -> step is ", step)
                # print("\t[PEAK WDW] valid f -> transformed scores is ", scores[f]+step)
                # print()
                #TODO limit to 0 or allow negatives?
                transform_scores.append(max(scores[f], scores[f] + step))
            else:
                # print("\t[PEAK WDW] f not in valid wdw range")
                transform_scores.append(scores[f])
                
        transform_scores_d[lbl] = transform_scores
    return transform_scores_d

def delta_multiplier(sims_d, delta_d):
    mul_scores_d = {}
    for lbl, scores in sims_d.items():
        mul_scores = []
        deltas = delta_d[lbl]
        d_avg = np.average(deltas)
        for x, m in zip(scores, deltas):
            mul_scores.append((m/d_avg)*x)
        mul_scores_d[lbl] = mul_scores
    return mul_scores_d
    

def center_on_avg_err(sims_d):
    mul_scores_d = {}
    for lbl, scores in sims_d.items():
        mul_scores = []
        s_avg = np.average(scores)
        for x in scores:
            mul_scores.append(x-s_avg)
        mul_scores_d[lbl] = mul_scores
    return mul_scores_d

#TODO
def score_combine(scores_d, sims_d, deltas_d):
    
    #consider multiplying sims_d by avg_delta/delta_i, therefore we amplify for the smaller scores -> that would have to happen in relative_scale
    sims = sims_d
    #SIMS
    sims = relative_scale(sims, 150, 0.5)
    # sims = delta_multiplier(sims, deltas_d)
    sims = peak_window(sims, 20)
    sims = center_on_avg_err(sims, deltas_d)
    sims = bulkrangeto1(sims, 2, 0)
    
    #MIL
    scores = scores_d    
    #LKKM
    
    #COMBINED


    # combine_d = {}
    #TODO rmember that scores_d and sims_d may be different sizes
    #TODO fix tmp -> ust use lbl when we have correct keys in annMIL data (predictions)
    # for lbl, sims in sims_d.items():
        # cat, num = lbl.split("/")
        # tmp = "/home/tomsch/Honours/Honours_Project/output/all-fc6/anomaly-videos/{}_Test/{}".format(cat, num) 
        # scores = scores_d[tmp]
        # np.add


    #peak window in here
    return sims, sims, scores_d
                       
   
def anno_vs_score_eval(antn, scores_d):
    collect_cm = {}
    collect_roc = {}
    
    for thresh in range(int(10*min_thresh), int(10*max_thresh+1)):
        anom_instances = []
        norm_instances = []
        anom_res = []
        norm_res = []
        
        anno_sgm_d = {}
        for lbl, scores in scores_d.items(): #TODO note this is extremely dependent on keys in dict
            scores = [float(i) for i in scores]
            cat, num = lbl.split('/')
            # print("\n[PRED_EVAL] class:{} - video:{}".format(cat, num))
            # /home/tomsch/Honours/Honours_Project/output/all-fc6/anomaly-videos/Burglary_Test/076
            if not "Normal" in cat:
                # print("\t[PRED EVAL] anomalous video")
                vid_anno = antn[antn[:, 0] == cat]
                vid_anno = np.squeeze(vid_anno[vid_anno[:, 1] == num])
                # print(vid_anno)
                if len(vid_anno) == 0:
                    pass
                    # print("\t\t************************WARNING: annotation not found for {}/{}".format(cat, num))
                else:
                    n_frames = int(vid_anno[2])
                    anno_cum = []
                    res = False
                    for pt in range(3, len(vid_anno), 2):
                        if vid_anno[pt] != '-1':
                            start_anom = int(vid_anno[pt])
                            end_anom = int(vid_anno[pt+1])
                            res, anno_sgm = verify_anom_fc6(scores, n_frames, ((start_anom, end_anom)), float(thresh/10), buffer=50, prev_res=res)   
                            
                            anno_cum += anno_sgm
                            anom_instances.append(1) #every anom is an instance even if 2 anoms come from same video
                            if res: 
                                anom_res.append(1) 
                            else:
                                anom_res.append(0)
                        else:
                            # print("-1 found, skip next -1")
                            pass
                    anno_sgm_d[lbl] = anno_cum
                    
            else:
                #TODO when we have normal videos
                # print("\t[PRED EVAL] normal video")
                norm_instances.append(1)
                print(float(thresh/10))
                violating = list(filter(lambda x: (x > float(thresh/10)), scores)) #maybe use filter instead
                if (len(violating) == 0):
                    norm_res.append(1) 
                else: 
                    norm_res.append(0)
                anno_sgm_d[lbl] = np.zeros(len(scores)).tolist()


            #pos is anomalous
            #neg is normal    
            
        tp = sum(anom_res)/sum(anom_instances)
        fn = 1 - tp
        tn = sum(norm_res)/sum(norm_instances) #TODO change back to normal instances
        # tn = float(thresh/10)
        fp = 1 - tn
        cm = np.asarray([[tp, fn],[fp, tn]])
        collect_cm[float(thresh/10)] = cm
        
        # instances = anom_instances + norm_instances
        # results = anom_res + norm_res
        
        tpr = tp/(tp+fn)
        fpr = fp/(tn+fp)
        collect_roc[float(thresh/10)] = (tpr, fpr)
    
    #calculate results at best threshold (for display of S or F marker)
    best_thresh = 0
    best_cm_sum = 0
    for t, mat in collect_cm.items():
        cm_sum = mat[0][0] + mat[1][1]
        if cm_sum > best_cm_sum:
            best_cm_sum = cm_sum
            best_thresh = t 
        
    print("BEST THRESHOLD: {}\ngetting result markers at that thresh ...".format(best_thresh))
    res_d = {}
    for lbl, scores in scores_d.items(): #TODO note this is extremely dependent on paths of file system (path must contain "anom" or "norm")
        scores = [float(i) for i in scores]
        cat, num = lbl.split('/')
        print("\n[PRED_EVAL] class:{} - video:{}".format(cat, num))
        # /home/tomsch/Honours/Honours_Project/output/all-fc6/anomaly-videos/Burglary_Test/076
        if not "Normal" in cat:
            print("\t[PRED EVAL] anomalous video")
            vid_anno = antn[antn[:, 0] == cat]
            vid_anno = np.squeeze(vid_anno[vid_anno[:, 1] == num])
            print(vid_anno)
            if len(vid_anno) == 0:
                print("\t\t************************WARNING: annotation not found for {}/{}".format(cat, num))
                res_d[lbl] = None
            else:
                n_frames = int(vid_anno[2])
                res = False
                for pt in range(3, len(vid_anno), 2):
                    if vid_anno[pt] != '-1':
                        start_anom = int(vid_anno[pt])
                        end_anom = int(vid_anno[pt+1])
                        res, anno_sgm = verify_anom_fc6(scores, n_frames, ((start_anom, end_anom)), best_thresh, buffer=50, prev_res=res)   
                    else:
                        # print("-1 found, skip next -1")
                        pass
                
                res_d[lbl] = res
        else:
            #TODO when we have normal videos
            print("\t[PRED EVAL] normal video")
            norm_instances.append(1)
            print(float(thresh/10))
            violating = list(filter(lambda x: (x > float(thresh/10)), scores)) #maybe use filter instead
            if (len(violating) == 0): 
                norm_res.append(1) 
            else:
                norm_res.append(0)
            res_d[lbl] = (len(violating)==0)

    
    plt.rcParams['text.color'] = "white"
    plt.rcParams['axes.labelcolor'] = "white"
    plt.rcParams['xtick.color'] = "white"
    plt.rcParams['ytick.color'] = "white"
    plt.rcParams['axes.facecolor']='0E1117'
    plt.rcParams['savefig.facecolor']='0E1117'

    
    cnt = 0
    r = 0
    n = int(10*(max_thresh-min_thresh)+1)
    figcm, axcm = plt.subplots(3,3,constrained_layout=True)
    # figcm.constrained_layout()
    for t, cmat in collect_cm.items():
        disp = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=CLASSES)
        disp.plot(cmap='plasma', ax=axcm[r][cnt % 3], colorbar=False) 
        axcm[r][cnt%3].title.set_text("threshold = {}".format(t))
        cnt+=1
        if cnt % 3 == 0:
            r += 1
            
      
    figroc, axroc = plt.subplots(1)
    tprs = [x[0] for i, x in collect_roc.items()]
    fprs = [x[1] for i, x in collect_roc.items()]
    axroc.plot(fprs, tprs)
    axroc.set_title("ROC Curve")  
    axroc.set_xlabel("False Positive Rate")
    axroc.set_ylabel("True Positive Rate")
    

    # fpr = {}
    # tpr = {}
    # roc_auc = {}
    # for i, s in collect_roc.items():
        # fpr[i], tpr[i], _ = roc_curve(s[0], s[1])
        # roc_auc[i] = auc(fpr[i], tpr[i])    
    
    # plt.show()
    
    # fig, ax = plt.subplots(1, 1)
    # disp = ConfusionMatrixDisplay(confusion_matrix=collect_cm[0.5], display_labels=CLASSES)
    # disp.plot(cmap='plasma', ax=ax, colorbar=False) 

    #TODO calc ROC and AUC and return as well
    return figcm, figroc, anno_sgm_d, res_d

def st_demo(combine_d, scores_d, sims_d, anno_sgm_d, res_d, cm_disp, roc_disp):
    
    # page_bg_img = '''
    #     <style>
    #     body {
    #     background-image: url("https://drive.google.com/file/d/1o-mnrINtNwuyfIgqChEIMrZX2-3Po_t9/view?usp=sharing");
    #     background-size: cover;
    #     }
    #     </style>
    #     '''
    # set_png_as_page_bg('/home/tomsch/Honours/Honours_Project/mn6-anomaly-detection/annMIL/misc/blurcctv.png')
    # st.markdown(page_bg_img, unsafe_allow_html=True)
    
    # st.title("  Anomaly Detection Framework")

    st.markdown("<h1 style='text-align: center; color: white; font-size: 1.5vw'>Anomaly Detection Framework</h1>", unsafe_allow_html=True)
    st.markdown("### This demo displays results of the application of an anomaly detection framework to a set of unseen CCTV surveillance videos.")
    st.markdown("### The framework assigns an anomaly score in the range [0, 1] per 16 frames of video.")
    st.markdown("### The final score is decided as a consensus between 3 models/heuristics, namely: *MIL base-model*, *CRAFT-flow*, *LKKM-flow*")
    st.markdown("### At each video, the display of relevant scores can be toggled and an option to view the video is presented.")
    st.markdown("### The categories of video include:")
    """
    * Abuse
    * Arrest
    * Arson
    * Assault
    * Burgalry
    * Explosion
    * Fighting
    * Road Accidents 
    * Robbery
    * Shooting
    * Shoplifitng
    * Stealing
    * Vandalism
    * Normal
    """
    
    st.write(roc_disp)
    st.write(cm_disp)
    "## Click on a video to view evaluations of the trained model on unseen footage."
    
    cnt = 0
    # print(sims_d.keys())
    
    # for vid, scores in scores_d.items():
    for vid in sorted(os.listdir(VIDEO_PATH)):
        # print("vid ---> ", vid)
        cnt += 1
        if vid.endswith(".mp4"):
            title = os.path.splitext(vid)[0][:-5]
            # print("title ---> ", title)
            
            if title[:6] == "Normal":
                cat = "Normal"
            else:
                cat = title[:-3]
            num = title[-3:]


            lbl = "{}/{}".format(cat, num)
            tmp = "/home/tomsch/Honours/Honours_Project/output/all-fc6/anomaly-videos/{}_Test/{}".format(cat, num) 
            
            # print(lbl)
            
            if lbl in res_d.keys():
                if res_d[lbl] is None:
                    emj = ":white_circle:" 
                else:
                    emj = ":large_blue_circle:" if res_d[lbl] else ":red_circle:" 
            
            exp = st.beta_expander(label=lbl, expanded=False)
            st.write(emj) 
            with exp:
                if exp.button("Playback Video", key=lbl):
                    if cat == "Normal":
                        path_mp4 = os.path.join(VIDEO_PATH, "{}_Videos_{}_x264.mp4".format(cat,num))
                    else:
                        path_mp4 = os.path.join(VIDEO_PATH, "{}{}_x264.mp4".format(cat,num))
                           
                    vf = open(path_mp4, "rb")
                    vb = vf.read()
                    st.video(vb)
                    
                # if lbl in scores_d.keys():
                # TODO should be using lbl which is just cat/num, replace tmp with lbl when pred-fc6.json follows key form of cat/num
                
                if lbl in combine_d.keys():
                    # print("is in combine_d.keys()******")
                    combine_data = pd.DataFrame({
                        "final scores": combine_d[lbl],
                    })
                    # all_data = pd.concat([all_data, flow_data], axis=1, ignore_index=True)
                else:
                    combine_data = pd.DataFrame({
                        "final scores": np.zeros(1), 
                    })
                    print("[PRED EVAL] <WARNING> no matching label for this video in COMBINE dict.")
                
                if lbl in sims_d.keys():
                    # print("is in sims_d.keys()******")
                    flow_data = pd.DataFrame({
                        "flow scores": sims_d[lbl],
                    })
                    # all_data = pd.concat([all_data, flow_data], axis=1, ignore_index=True)
                else:
                    flow_data = pd.DataFrame({
                        "flow scores": np.zeros(1), 
                    })
                    print("[PRED EVAL] <WARNING> no matching label for this video in FLOW dict.")
                    flow_data.rename("MIL scores")
                    
                # tmp = "/home/tomsch/Honours/Honours_Project/output/all-fc6/anomaly-videos/{}_Test/{}".format(cat, num) 
                if lbl in scores_d.keys():
                    mil_data = pd.DataFrame({
                        "MIL scores": scores_d[lbl],
                        # "MIL scores": np.zeros(10), 
                    })
                else:
                    mil_data = pd.DataFrame({
                        # "MIL scores": scores_d[tmp],
                        "MIL scores": np.zeros(1), 
                    })
                    print("[PRED EVAL] WARNING no matching label for this video in MIL scores dict.")    
                    
                res = None
                anno_dat = np.zeros(len(sims_d[lbl])).tolist()
                if lbl in anno_sgm_d.keys():
                    # print("is in anno_sgm.keys()*********")
                    anno_sgms = anno_sgm_d[lbl]
                    # anno_dat = np.squeeze(anno_dat)
                    if cat != "Normal":
                        # print("anno_sgms")
                        # print(anno_sgms)
                        for a in anno_sgms:
                           anno_dat[int(a)] = 1 #1 indexing

                        # print(anno_dat)
                        set1 = False
                        for a in range(len(anno_dat)):
                            # print(a, end=" ")
                            if not set1:
                                if anno_dat[int(a)] == 1:
                                    set1 = True
                            else:
                                if anno_dat[int(a)] == 1:
                                    set1 = False
                                else:
                                    anno_dat[int(a)] = 1
                        # print(anno_dat)
                        # print(len(anno_dat))
                    else:
                        anno_dat[0] = 1 #to ensure that y-ax is scaled to 1 for normal footage
                        
                        
                    # sanitize_len = max(sanitize_len, len(anno_dat))
                    anno_data = pd.DataFrame({
                        "annotations": anno_dat,
                    })
                    # all_data = pd.concat([all_data, anno_data], axis=1, ignore_index=True)              
                else:
                    anno_data = pd.DataFrame({
                        "annotations": anno_dat,
                    })
                    print("[PRED EVAL] <WARNING> no matching label for this video in ANNO dict.")
                
                all_data = anno_data 
                # indices = []
                # indices.append("annotations")
                # all_data["vid"] = "{}{}".format(cat, num)
                # all_data["anno"] = anno_data
                # all_data["MIL"] = mil_data
                # all_data["FLOW"] = flow_data

                st.checkbox("display MIL", value=False, key="{}{}MIL".format(cat, num)) 
                st.checkbox("display FLOW", value=True, key="{}{}FLOW".format(cat, num))
                st.checkbox("display FINAL", value=False, key="{}{}FINAL".format(cat, num))
    
                if eval("st.session_state.{}{}MIL".format(cat, num)):
                    all_data = pd.concat([all_data, mil_data], axis=1, ignore_index=True)              
                    # indices.append("MIL base scores")
                if eval("st.session_state.{}{}FLOW".format(cat, num)):
                    all_data["CRAFT scores"] = flow_data
                    # all_data = pd.concat([all_data, flow_data], axis=1, ignore_index=True)              
                    # indices.append("CRAFT scores")

                if eval("st.session_state.{}{}FINAL".format(cat, num)):
                    all_data = pd.concat([all_data, combine_data], axis=1, ignore_index=True)              
                    # indices.append("combined scores")
                # all_data.columns = indices
        
                st.area_chart(all_data, use_container_width=True, )


        
def main(pred_path, sim_path, delta_path):
    antn = read_anno_file(ANNO_FILE)
    with open(pred_path, "r") as f:
        scores = json.load(f)
        scores_d = json.loads(scores)
    with open(sim_path, "r") as f:
        sims = json.load(f)
        sims_d = json.loads(sims)
    with open(delta_path, "r") as f:
        deltas = json.load(f)
        deltas_d = json.loads(deltas)
    
    # combine_d, sims_d, scores_d = score_combine(scores_d, sims_d, deltas_d)

    #DOD2: for display of CRAFT/sims_d potential
    sims_d = individual_scale(sims_d, 1)
    sims_d = center_on_avg_err(sims_d)
    sims_d = rangeto1(sims_d)
    
    
    #confusion matrix and ROC/AUC
    cm_disp, roc_disp, anno_sgm_d, res_d = anno_vs_score_eval(antn, sims_d)
    st_demo(combine_d=sims_d, scores_d=scores_d, sims_d=sims_d, anno_sgm_d=anno_sgm_d, res_d=res_d, cm_disp=cm_disp, roc_disp=roc_disp)
        
if __name__ == '__main__':
    pred_path = sys.argv[1]
    sim_path = sys.argv[2]
    delta_path = sys.argv[3]
    
    main(pred_path, sim_path, delta_path)
    
    
    
#@st.cache(allow_output_mutation=True)
#def get_base64_of_bin_file(bin_file):
#    with open(bin_file, 'rb') as f:
#        data = f.read()
#    return base64.b64encode(data).decode()
#
#def set_png_as_page_bg(png_file):
#    bin_str = get_base64_of_bin_file(png_file)
#    page_bg_img = '''
#    <style>
#    body {
#    background-image: url("data:image/png;base64,%s");
#    background-size: cover;
#    }
#    </style>
#    ''' % bin_str
#    
#    st.markdown(page_bg_img, unsafe_allow_html=True)
#    return
#
#
#def plot_data(**kwargs):
#    # if disp:
#    # if "{}-MIL".format(kwargs["vid"]) in st.session_state:
#    # st.write("FOUND IT")
#    
#    all_data = kwargs["anno"]
#    
#    if eval("st.session_state.{}MIL".format(kwargs["vid"])):
#        all_data = pd.concat([all_data, kwargs["MIL"]], axis=1, ignore_index=True)              
#    if eval("st.session_state.{}FLOW".format(kwargs["vid"])):
#        all_data = pd.concat([all_data, kwargs["FLOW"]], axis=1, ignore_index=True)              
#        
#    return st.area_chart(all_data, use_container_width=True)
#
#    
#def relative_scaling(sims, deltas):
#    scaled_sims = {}
#    
#    # SIM ERR PER DELTA -> POOR RESULTS
#    # for lbl, sim in sims.items():
#        # sc_sim = []
#        # delta = deltas[lbl]
#        # ad = np.average(delta)
#        # for s, d in zip(sim, delta):
#            # sc_sim.append(s/(d+ad))
#        # scaled_sims[lbl] = sc_sim
#    # return scaled_sims    
#    print("[REL SCALE] len of sims in: ", len(sims))
#    # sim_keys = sims.keys()
#    # shuffle(sim_keys)
#    batch_size = int(len(sims)/10)    
#    for batch in range(1, 10):
#        sc = StandardScaler()
#        
#        if batch == 9:
#            subset = islice(sims.items(), int((batch-1)*batch_size), None)
#        else:
#            subset = islice(sims.items(), int((batch-1)*batch_size), int(batch*batch_size))
#        
#        batch_values = []
#        for lbl, sim in subset:
#            # print(lbl)
#            # sc.partial_fit(np.expand_dims(sim, axis=0))
#            batch_values.extend(sim)
#        # print("len batch values: ", len(batch_values))
#        sc.fit(np.asarray(batch_values).reshape((1, len(batch_values))))
#        print(sc.mean_)
#        print(sc.scale_)
#        
#        u = sc.mean_
#        s = sc.scale_
#        
#        if batch == 9:
#            subset = islice(sims.items(), int((batch-1)*batch_size), None)
#        else:
#            subset = islice(sims.items(), int((batch-1)*batch_size), int(batch*batch_size))
#        
#        for lbl, sim in subset:
#            sim_sc = [(x-u)/s for x in sim]
#            # print("initial sim len: ", len(sim))
#            l = len(sim_sc)
#            print(sim_sc)
#            scaled_sims[lbl] = np.asarray(sim_sc).reshape(l, 1)
#            # print("scaled sim len: ", len(scaled_sims[lbl]))
#            
#    print("[REL SCALE] len of sims out: ", len(scaled_sims))
#    time.sleep(10)
#    
#    return scaled_sims
#
##in case of 32 sgm verification
##DEPRECATED - using all fc6 verfification rather
#def verify_anom(scores_32, n_frm, anom_window, threshold):
#    n_frames = n_frm - 200 #TODONOTE trim acocunted for here
#    n_frames_sgm = int(n_frames/32)
#    frame_borders = [border for border in range(0, n_frames, n_frames_sgm)]
#    frame_borders = frame_borders[:32]
#    annotated_sgms = []
#    for i in range(0, len(frame_borders)):
#        if anom_window[0] >= frame_borders[i] and anom_window[0] <= (frame_borders[i])+n_frames_sgm:
#            annotated_sgms.append(i)
#        if anom_window[1] >= frame_borders[i] and anom_window[1] <= (frame_borders[i])+n_frames_sgm:
#            annotated_sgms.append(i)
#
#    #check these indices' anom scores
#    uniq_anno_sgms = list(set(annotated_sgms)) #get unique list i.e., remove doubles
#    uniq_anno_sgms = sorted(uniq_anno_sgms)
#    print("\t\t[PRED_EVAL] 0-indexed annotation segments: {}".format(uniq_anno_sgms))
#    print("\t\t[PRED_EVAL] number of segment scores: {}".format(len(scores_32)))
#    for anno in range(uniq_anno_sgms[0], (uniq_anno_sgms[-1:][0])+1, 1):
#        pred_score = scores_32[anno]
#        print("\t\t\t1-indexed annotated sgm {} - has predicted score of {}".format(anno+1, pred_score))
#        #if any sgm which falls within annotated window has anom_score above threshold - true pos for anomaly detec
#        if pred_score > threshold:
#            return True, uniq_anno_sgms
#    
#    return False, uniq_anno_sgms
#            
#    #annotations are made with 30fps fixed
#    