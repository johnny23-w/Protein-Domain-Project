import pandas as pd
import numpy as np
import torch

def probs_to_doms(probs):
    preds = probs > 0.5
    counting = False
    c_dom_start = None
    c_dom_end = None
    dom_list = []
    for idx, pred in enumerate(preds):
        if pred == 0:
            if not counting:
                c_dom_start = idx + 1
                c_dom_end = idx + 1
                counting = True
            else:
                c_dom_end = idx + 1
        else:
            if counting:
                dom_list.append([c_dom_start, c_dom_end])
                counting = False
                c_dom_start = None
                c_dom_end = None
    if counting:
        dom_list.append([c_dom_start, c_dom_end])
    return [dom for dom in dom_list if dom[1]-dom[0] > 30]

def parse_domain(dom_str):
    dom_list = [[it[0], list(map(int, it[1].split(':')))] for it in (dom.split(';') for dom in dom_str.split('|'))]
    return dom_list

def add_link_to_doms(doms, prot_len):
    prev_dom = None
    outp = []
    link_num = 1
    dom_num = 1
    for dom in doms:
        if prev_dom == None:
            if dom[0]!=1:
                outp.append(['l'+str(link_num),[1,dom[0]-1]])
            link_num += 1
            outp.append(['d'+str(dom_num),dom])
            dom_num += 1
            prev_dom = dom
        else:
            if dom[0]!=prev_dom[1]+1:
                outp.append(['l'+str(link_num),[prev_dom[1]+1,dom[0]-1]])
            outp.append(['d'+str(dom_num),dom])
            link_num+=1
            dom_num += 1
            prev_dom = dom
    if prev_dom[1]!=prot_len:
        outp.append(['l'+str(link_num),[prev_dom[1]+1,prot_len]])

    return outp

def calc_overlap(int1, int2):
    return max(0, min(int1[1], int2[1]) - max (int1[0], int2[0]) + 1)

def calculate_overlap_matrix(tar_list, pred_list):
    tar_dict = {k:v for (k,v) in tar_list}
    pred_dict = {k:v for (k,v) in pred_list}
    overlap_mat = pd.DataFrame(columns=tar_dict.keys(), index=pred_dict.keys())
    for tar_dom in tar_dict.keys():
        for pred_dom in pred_dict.keys():
            overlap_mat.loc[pred_dom, tar_dom] = calc_overlap(pred_dict[pred_dom], tar_dict[tar_dom])
    return overlap_mat

def consolidate_overlap_matrix(overlap_mat):
    overlap_mat_collapse_cols = pd.DataFrame(columns=[col for col in overlap_mat.columns if 'd' in col]+['l'], index=overlap_mat.index)
    for col in overlap_mat_collapse_cols:
        if 'd' in col:
            overlap_mat_collapse_cols.loc[:,col] = overlap_mat.loc[:,col]
        elif col=='l':
            overlap_mat_collapse_cols.loc[:,col] = overlap_mat.loc[:,[col for col in overlap_mat.columns if 'l' in col]].sum(axis=1)
    consol_mat = pd.DataFrame(columns=[col for col in overlap_mat.columns if 'd' in col]+['l'], index=[ind for ind in overlap_mat.index if 'd' in ind]+['l'])
    for row in consol_mat:
        if 'd' in row:
            consol_mat.loc[row,:] = overlap_mat_collapse_cols.loc[row,:]
        elif row=='l':
            consol_mat.loc[row,:] = overlap_mat_collapse_cols.loc[[ind for ind in overlap_mat_collapse_cols.index if 'l' in ind]].sum(axis=0)
    return consol_mat

def calc_ndo_from_consol(consol_mat, perf_score):
    col_scores = []
    row_scores = []
    for col in consol_mat.columns:
        if 'd' in col:
            col_scores.append((2*max(consol_mat.loc[:,col]))-((consol_mat.loc[:,col]).sum()))
    for ind in consol_mat.ind:
        if 'd' in ind:
            row_scores.append((2*max(consol_mat.loc[ind,:]))-((consol_mat.loc[ind,:]).sum()))
    return ((sum(col_scores) + sum(row_scores))/2)/perf_score

def num_dom_res(tar_list):
    num = 0
    dom_list = [dom for dom in tar_list if 'd' in dom[0]]
    for dom in dom_list:
        num += dom[1][1]-dom[1][0] + 1
    return num

def ndo(tar_list, pred_list):
    overlap_mat = calculate_overlap_matrix(tar_list=tar_list, pred_list=pred_list)
    consol_mat = consolidate_overlap_matrix(overlap_mat=overlap_mat)
    perf_score = num_dom_res(tar_list=tar_list)
    return calc_ndo_from_consol(consol_mat=consol_mat, perf_score=perf_score)