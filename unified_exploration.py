import pandas as pd
import numpy as np
import os
import pathlib
from graphtempo import *
import itertools
import copy
import time
from collections import defaultdict
from collections import OrderedDict
import collections
import ast
import plotly.express as px
import plotly.io as pio
#pio.renderers.default='browser'
pio.renderers.default = "svg"
import ast


# distinct AGGREGATION for PROPERTY GRAPHS | STATIC

def Intersection_Static(nodesdf,edgesdf,tia,intvl):
    # get intersection of nodes and edges on interval
    nodes_inx = nodesdf[intvl][nodesdf[intvl].all(axis=1)]
    edges_inx = edgesdf[intvl][edgesdf[intvl].all(axis=1)]
    tia_inx = tia[tia.index.isin(nodes_inx.index)]
    inx = [nodes_inx,edges_inx]
    return(inx,tia_inx)

def Intersection_Static_UN(nodesdf,edgesdf,tia,left_invl,right_invl):
    nodes_u = nodesdf[left_invl+right_invl][nodesdf[left_invl].any(axis=1)]
    n = nodes_u[nodes_u[right_invl].any(axis=1)]
    e = edgesdf[left_invl+right_invl][edgesdf[left_invl].any(axis=1)]
    e = e[e[right_invl].any(axis=1)]
    tiainx = tia[tia.index.isin(n.index)]
    ne = [n,e]
    return(ne,tiainx)

def Diff_Static(nodesdf,edgesdf,tia,intvl_fst,intvl_scd):
    un_init, tia_init = Union_Static(nodesdf,edgesdf,tia,intvl_fst)
    un_to_rm, tia_to_rm = Union_Static(nodesdf,edgesdf,tia,intvl_scd)
    nodes = un_init[0][~un_init[0].index.isin(un_to_rm[0].index)]
    edges = un_init[1][~un_init[1].index.isin(un_to_rm[1].index)]
    ediff_idx = set([item for i in edges.index.values.tolist() for item in i])
    #tia_d = tia_init[tia_init.index.isin(ediff_idx)]
    tia_d = tia_init[tia_init.index.get_level_values('id').isin(ediff_idx)]
    diff = [nodes,edges]
    return(diff,tia_d)


def Aggregate_Static_Dist_PG(inx,tia_inx,stc_attrs,node_type):
    #types_to_choose = list(inx[0].index.get_level_values('type').value_counts().index)
    if inx[0].index.equals(tia_inx.index):
        #nodes = tia_inx[stc_attrs].set_index(tia_inx[stc_attrs].columns.values.tolist())
        nodes = tia_inx[stc_attrs]
    else:#difference output produces different indexes for nodes and attributes
        nodes = pd.DataFrame(index=inx[0].index)
        for attr in stc_attrs:
            nodes[attr] = tia_inx.loc[nodes.index,attr].values
    
    nodes_orig_not_agg = set([i[0] for i in nodes.index if i[1] not in node_type])
    nodes = nodes.set_index(stc_attrs, append=True)
    idx_agg = [i for i in nodes.index if i[1] in node_type]
    idx_not_agg = [i for i in nodes.index if i[1] not in node_type]
    nodes_agg = nodes.loc[idx_agg]
    nodes_agg = nodes_agg.droplevel('id')
    nodes_agg = nodes_agg.groupby(nodes_agg.index.names).size().to_frame('count')
    nodes_not_agg = nodes.loc[idx_not_agg]
    nodes_not_agg = nodes_not_agg.droplevel(stc_attrs)
    nodes_not_agg = nodes_not_agg.swaplevel()
    nodes_not_agg['count'] = 1
    nodes_all = pd.concat([nodes_agg, nodes_not_agg], axis=0)
    
    # edges
    edges = pd.DataFrame(index=inx[1].index)
    edges_idx = edges.index.tolist()
    
    for attr in stc_attrs:
        edges[attr+'L'] = \
        tia_inx.loc[edges.index.get_level_values('Left'),attr].values
    for attr in stc_attrs:
        edges[attr+'R'] = \
        tia_inx.loc[edges.index.get_level_values('Right'),attr].values
    for attr in stc_attrs:
        for idx in edges_idx:
            if idx[0] in nodes_orig_not_agg:
                edges.loc[idx, attr+'L'] = idx[0]
            if idx[1] in nodes_orig_not_agg:
                edges.loc[idx, attr+'R'] = idx[1]
    edges = edges.droplevel(['Left','Right'])
    edges = edges.set_index(edges.columns.values.tolist(), append=True)
    edges = edges.groupby(edges.index.names).size().to_frame('count')
    agg = [nodes_all, edges]
    return(agg)


def Aggregate_Static_Dist_PG_UNDIRECTED(inx,tia_inx,stc_attrs,node_type):
    #types_to_choose = list(inx[0].index.get_level_values('type').value_counts().index)
    if inx[0].index.equals(tia_inx.index):
        #nodes = tia_inx[stc_attrs].set_index(tia_inx[stc_attrs].columns.values.tolist())
        nodes = tia_inx[stc_attrs]
    else:#difference output produces different indexes for nodes and attributes
        nodes = pd.DataFrame(index=inx[0].index)
        for attr in stc_attrs:
            nodes[attr] = tia_inx.loc[nodes.index,attr].values
    
    nodes_orig_not_agg = set([i[0] for i in nodes.index if i[1] not in node_type])
    nodes = nodes.set_index(stc_attrs, append=True)
    idx_agg = [i for i in nodes.index if i[1] in node_type]
    idx_not_agg = [i for i in nodes.index if i[1] not in node_type]
    nodes_agg = nodes.loc[idx_agg]
    nodes_agg = nodes_agg.droplevel('id')
    nodes_agg = nodes_agg.groupby(nodes_agg.index.names).size().to_frame('count')
    nodes_not_agg = nodes.loc[idx_not_agg]
    nodes_not_agg = nodes_not_agg.droplevel(stc_attrs)
    nodes_not_agg = nodes_not_agg.swaplevel()
    nodes_not_agg['count'] = 1
    nodes_all = pd.concat([nodes_agg, nodes_not_agg], axis=0)
    
    # edges
    edges = pd.DataFrame(index=inx[1].index)
    edges_idx = edges.index.tolist()
    
    for attr in stc_attrs:
        edges[attr+'L'] = \
        tia_inx.loc[edges.index.get_level_values('Left'),attr].values
    for attr in stc_attrs:
        edges[attr+'R'] = \
        tia_inx.loc[edges.index.get_level_values('Right'),attr].values
    for attr in stc_attrs:
        for idx in edges_idx:
            if idx[0] in nodes_orig_not_agg:
                edges.loc[idx, attr+'L'] = idx[0]
            if idx[1] in nodes_orig_not_agg:
                edges.loc[idx, attr+'R'] = idx[1]
    edges = edges.droplevel(['Left','Right'])
    edges = edges.set_index(edges.columns.values.tolist(), append=True)
    edges = edges.groupby(edges.index.names).size().to_frame('count')
    
    type_index_list = edges.index.tolist()
    type_index_dict = {i[1:]:i[0] for i in type_index_list}
    edges = edges.droplevel('type')

    edges_inx_names = edges.index.names
    edges_idx = edges.index.tolist()
    edges_idx_new = []
    for tpl in edges_idx:
        edges_idx_new.append(tuple([tpl[:int(len(tpl)/2)], tpl[int(len(tpl)/2):]]))
    edges_idx_new = [tuple(sorted(tuple([i[0],i[1]]))) for i in edges_idx_new]
    edges_idx_new = [tuple(tpl[0]+tpl[1]) for tpl in edges_idx_new]
    edges_idx_new = pd.MultiIndex.from_tuples(edges_idx_new)
    edges.index = edges_idx_new

    edges = edges.groupby(edges.index).sum()
    edges.index = pd.MultiIndex.from_tuples(edges.index.tolist())
    edges.index.names = edges_inx_names
    
    for key,val in type_index_dict.items():
        if key in edges.index:
            edges.loc[key, 'type'] = val
    edges = edges.set_index('type', append=True)
    agg = [nodes_all, edges]
    return(agg)




# 1. UNIFIED / ONE-PASS SKYLINE EXPLORATION
# SKYLINE ONE-PASS | STATIC | INX strict --> STABILITY


# =============================================================================
# def SkylinExpl1(intervals, attr_val_combs, edge_type, node_type, stc_attrs, event, nodesdf, edgesdf, directed):
#     skyline_st = []
#     dominate_counter = {}
#     for left,right in intervals:
#         print('left, right: ', left, right)
#         while len(left) >= 1:
#             current_w = []
#             if event == 'stability_strict':
#                 ev,tia_ev = Intersection_Static(nodesdf,edgesdf,time_invar,left+right)
#             if event == 'stability_loose':
#                 ev,tia_ev = Intersection_Static_UN(nodesdf,edgesdf,time_invar,left,right)
#             if event == 'growth_loose':
#                 ev,tia_ev = Diff_Static(nodesdf,edgesdf,time_invar,right,left)
#             elif event == 'shrinkage_loose':
#                 ev,tia_ev = Diff_Static(nodesdf,edgesdf,time_invar,left,right)
#             if not ev[1].empty:
#                 if directed == True:
#                     agg = Aggregate_Static_Dist_PG(ev,tia_ev,stc_attrs,node_type)
#                 else:
#                     agg = Aggregate_Static_Dist_PG_UNDIRECTED(ev,tia_ev,stc_attrs,node_type)
#                 edges_sky = agg[1][agg[1].index.get_level_values('type').isin(edge_type)].droplevel('type')
#                 if not edges_sky.empty:
#                     edges_sky_idx = [tuple([i for i in tpl if i!='0']) for tpl in edges_sky.index]
#                     edges_sky.index = pd.MultiIndex.from_tuples(edges_sky_idx)
#                     for comb in attr_val_combs:
#                         if comb in edges_sky_idx:
#                             result = edges_sky.loc[comb][0]
#                         else:
#                             result = 0
#                         current_w.append(result)
#                     if all([w==0 for w in current_w]):
#                         left = left[1:]
#                         continue
#                     current_w.append(len(left))
#                     dominate_counter[str((current_w,left,right))] = 0
#                     if not skyline_st:
#                         skyline_st.append([current_w,left,right])
#                     else:
#                         flags = []
#                         sky_rm = []
#                         for sky in skyline_st:
#                             if all([current_w[i] >= sky[0][i] for i in range(len(current_w))]) and \
#                                 any([current_w[i] > sky[0][i] for i in range(len(current_w))]):
#                                 dominate_counter[str((current_w,left,right))] += 1
#                                 dominate_counter[str((current_w,left,right))] += dominate_counter[str(tuple(sky))]
#                                 dominate_counter[str(tuple(sky))] = 0
#                                 sky_rm.append(sky)
#                                 flags.append(1)
#                             # tie
#                             elif any([current_w[i] > sky[0][i] for i in range(len(current_w))]) or \
#                                 all([current_w[i] == sky[0][i] for i in range(len(current_w))]):
#                                 flags.append(2)
#                             # curr dominated by sky
# # =============================================================================
# #                             elif all([current_w[i] <= sky[0][i] for i in range(len(current_w))]) and \
# #                                 any([current_w[i] < sky[0][i] for i in range(len(current_w))]):
# # =============================================================================
#                             else:
#                                 dominate_counter[str(tuple(sky))] += 1
#                                 flags.append(3)
#                         skyline_st = [sky for sky in skyline_st if sky not in sky_rm]
#                         if 3 not in flags:
#                             skyline_st.append([current_w,left,right])
#             left = left[1:]
#     return(skyline_st, dominate_counter)
# =============================================================================


def SkylinExpl1(intervals, attr_val_combs, edge_type, node_type, stc_attrs, event, nodesdf, edgesdf, directed):
    skyline_st = []
    dominate_counter = {}
    for left,right in intervals:
        #print('left, right: ', left, right)
        while len(left) >= 1:
            current_w = []
            if event == 'stability_strict':
                ev,tia_ev = Intersection_Static(nodesdf,edgesdf,time_invar,left+right)
            if event == 'stability_loose':
                ev,tia_ev = Intersection_Static_UN(nodesdf,edgesdf,time_invar,left,right)
            if event == 'growth_loose':
                ev,tia_ev = Diff_Static(nodesdf,edgesdf,time_invar,right,left)
            elif event == 'shrinkage_loose':
                ev,tia_ev = Diff_Static(nodesdf,edgesdf,time_invar,left,right)
            if not ev[1].empty:
                if directed == True:
                    agg = Aggregate_Static_Dist_PG(ev,tia_ev,stc_attrs,node_type)
                else:
                    agg = Aggregate_Static_Dist_PG_UNDIRECTED(ev,tia_ev,stc_attrs,node_type)
                edges_sky = agg[1][agg[1].index.get_level_values('type').isin(edge_type)].droplevel('type')
                if not edges_sky.empty:
                    edges_sky_idx = [tuple([i for i in tpl if i!='0']) for tpl in edges_sky.index]
                    edges_sky.index = pd.MultiIndex.from_tuples(edges_sky_idx)
                    for comb in attr_val_combs:
                        if comb in edges_sky_idx:
                            result = edges_sky.loc[comb][0]
                        else:
                            result = 0
                        current_w.append(result)
                    if all([w==0 for w in current_w]):
                        left = left[1:]
                        continue
                    if event == 'stability_strict' or event == 'growth_loose':
                        current_w.append(len(left))
                    else:
                        current_w.append(-len(left))
                    dominate_counter[str((current_w,left,right))] = 0
                    if not skyline_st:
                        skyline_st.append([current_w,left,right])
                    else:
                        flags = []
                        sky_rm = []
                        for sky in skyline_st:
                            if all([current_w[i] >= sky[0][i] for i in range(len(current_w))]) and \
                                any([current_w[i] > sky[0][i] for i in range(len(current_w))]):
                                dominate_counter[str((current_w,left,right))] += 1
                                dominate_counter[str((current_w,left,right))] += dominate_counter[str(tuple(sky))]
                                dominate_counter[str(tuple(sky))] = 0
                                sky_rm.append(sky)
                                flags.append(1)
                            # tie
                            elif any([current_w[i] > sky[0][i] for i in range(len(current_w))]) or \
                                all([current_w[i] == sky[0][i] for i in range(len(current_w))]):
                                flags.append(2)
                            # curr dominated by sky
                            else:
                                dominate_counter[str(tuple(sky))] += 1
                                flags.append(3)
                        skyline_st = [sky for sky in skyline_st if sky not in sky_rm]
                        if 3 not in flags:
                            skyline_st.append([current_w,left,right])
            left = left[1:]
    return(skyline_st, dominate_counter)

# counting time for agg and skylines separately
def SkylinExpl(intervals, attr_val_combs, edge_type, node_type, stc_attrs, event, nodesdf, edgesdf, directed):
    skyline_st = []
    dominate_counter = {}
    agg_time = 0
    for left,right in intervals:
        #print('left, right: ', left, right)
        while len(left) >= 1:
            current_w = []
            start_agg = time.perf_counter()
            if event == 'stability_strict':
                ev,tia_ev = Intersection_Static(nodesdf,edgesdf,time_invar,left+right)
            elif event == 'stability_loose':
                ev,tia_ev = Intersection_Static_UN(nodesdf,edgesdf,time_invar,left,right)
            elif event == 'growth_loose':
                ev,tia_ev = Diff_Static(nodesdf,edgesdf,time_invar,right,left)
            elif event == 'shrinkage_loose':
                ev,tia_ev = Diff_Static(nodesdf,edgesdf,time_invar,left,right)
            if not ev[1].empty:
                if directed == True:
                    agg = Aggregate_Static_Dist_PG(ev,tia_ev,stc_attrs,node_type)
                else:
                    agg = Aggregate_Static_Dist_PG_UNDIRECTED(ev,tia_ev,stc_attrs,node_type)
                end_agg = time.perf_counter()
                edges_sky = agg[1][agg[1].index.get_level_values('type').isin(edge_type)].droplevel('type')
                if not edges_sky.empty:
                    edges_sky_idx = [tuple([i for i in tpl if i!='0']) for tpl in edges_sky.index]
                    edges_sky.index = pd.MultiIndex.from_tuples(edges_sky_idx)
                    for comb in attr_val_combs:
                        if comb in edges_sky_idx:
                            result = edges_sky.loc[comb][0]
                        else:
                            result = 0
                        current_w.append(result)
                    if all([w==0 for w in current_w]):
                        left = left[1:]
                        continue
                    if event == 'stability_strict' or event == 'growth_loose':
                        current_w.append(len(left))
                    else:
                        current_w.append(-len(left))
                    dominate_counter[str((current_w,left,right))] = 0
                    if not skyline_st:
                        skyline_st.append([current_w,left,right])
                    else:
                        flags = []
                        sky_rm = []
                        for sky in skyline_st:
                            if all([current_w[i] >= sky[0][i] for i in range(len(current_w))]) and \
                                any([current_w[i] > sky[0][i] for i in range(len(current_w))]):
                                dominate_counter[str((current_w,left,right))] += 1
                                dominate_counter[str((current_w,left,right))] += dominate_counter[str(tuple(sky))]
                                dominate_counter[str(tuple(sky))] = 0
                                sky_rm.append(sky)
                                flags.append(1)
                            # tie
                            elif any([current_w[i] > sky[0][i] for i in range(len(current_w))]) or \
                                all([current_w[i] == sky[0][i] for i in range(len(current_w))]):
                                flags.append(2)
                            # curr dominated by sky
                            else:
                                dominate_counter[str(tuple(sky))] += 1
                                flags.append(3)
                        skyline_st = [sky for sky in skyline_st if sky not in sky_rm]
                        if 3 not in flags:
                            skyline_st.append([current_w,left,right])
            else:
                end_agg = time.perf_counter()
            agg_time += end_agg - start_agg
            left = left[1:]
    return(skyline_st, dominate_counter, agg_time)


# Read DBLP

nodes_df = pd.read_csv('../skyline_extension/datasets/DBLP/dblp_prop_graph/nodes_df.csv', sep=' ', index_col=[0,1])
edges_df = pd.read_csv('../skyline_extension/datasets/DBLP/dblp_prop_graph/edges_df.csv', sep=' ', index_col=[0,1,2])
time_invar = pd.read_csv('../skyline_extension/datasets/DBLP/dblp_prop_graph/time_invar.csv', sep=' ', index_col=[0,1], dtype=str)
time_var = pd.read_csv('../skyline_extension/datasets/DBLP/dblp_prop_graph/time_var.csv', sep=' ', index_col=[0,1], dtype=str)


c=0
intvls = []
for i in range(1,len(edges_df.columns)+1-c):
    intvls.append([list(edges_df.columns[:i]), list(edges_df.columns[i:i+1])])
    c += 1
intvls = intvls[:-1]


# increasing dataset
nodes_sliced = [nodes_df.iloc[:,:6], nodes_df.iloc[:,:11], nodes_df.iloc[:,:16], nodes_df.iloc[:,:21]]
nodes_sliced = [i.loc[~(i==0).all(axis=1)] for i in nodes_sliced]
edges_sliced = [edges_df.iloc[:,:6], edges_df.iloc[:,:11], edges_df.iloc[:,:16], edges_df.iloc[:,:21]]
edges_sliced = [i.loc[~(i==0).all(axis=1)] for i in edges_sliced]
dataset_sliced = {'nodes': nodes_sliced, 'edges': edges_sliced}
intvls_sliced = [intvls[:5], intvls[:10], intvls[:15], intvls[:20]]

slices = len(intvls_sliced)


attr_val_left = time_invar.gender[time_invar.index.get_level_values('type') == 'author'].unique().tolist()
#attr_val_right = time_invar[time_invar.index.get_level_values('type') == 'conference'].index.get_level_values('id').tolist()
attr_val_right = time_invar.topic[time_invar.index.get_level_values('type') == 'conference'].unique().tolist()

attr_val_combs_coll = list(itertools.product(attr_val_left, attr_val_left))
attr_val_combs_publ = list(itertools.product(attr_val_left, attr_val_right))
attr_val_combs_coll_publ = copy.deepcopy(attr_val_combs_coll+attr_val_combs_publ)


# DBLP EXPERIMENTS
#1a
# skyline exploration for agg by NODE: author.gender AND EDGE: collaborate
total_times = []
event_col = []
total_duration_avg = []
total_duration_agg_avg = []
dominations_dblp_colla = {}
skylines_dblp_colla = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    duration = []
    duration_agg = []
    for j in range(2):
        duration_of_slices = []
        duration_of_slices_agg = []
        for d in range(slices):
            start = time.perf_counter()
            sky,dom,t_agg = SkylinExpl(intvls_sliced[d], attr_val_combs=copy.deepcopy(attr_val_combs_coll),
                          edge_type=['collaboration'], node_type=['author'],
                          stc_attrs=['gender'], event=i, nodesdf=dataset_sliced['nodes'][d],
                          edgesdf=dataset_sliced['edges'][d], directed=True)
            end = time.perf_counter()
            duration_of_slices.append(end - start)
            duration_of_slices_agg.append(t_agg)
            if d == slices-1:
                dominations_dblp_colla[i] = dom
                skylines_dblp_colla[i] = sky
        duration.append(duration_of_slices)
        duration_agg.append(duration_of_slices_agg)
    duration = pd.DataFrame(duration).T
    duration_avg = list(duration.mean(axis=1))
    duration_agg = pd.DataFrame(duration_agg).T
    duration_agg_avg = list(duration_agg.mean(axis=1))
    total_duration_avg.extend(duration_avg)
    total_duration_agg_avg.extend(duration_agg_avg)
    interval_col = ['up to ' + intvls_sliced[d][-1][-1][0] for d in range(slices)]
    duration['interval'] = interval_col
    total_times.append(duration)
    event_col.extend([i]*slices)

total_times_colla = pd.concat(total_times)
total_times_colla['time_total'] = total_duration_avg
total_times_colla['time_agg'] = total_duration_agg_avg
total_times_colla['time_sky'] = total_times_colla['time_total'] - total_times_colla['time_agg']
total_times_colla['event'] = event_col
total_times_colla = total_times_colla.round(2)
total_times_colla.to_csv('exp_results/dblp/author.gender_edge.coll.csv', header='column_names')
total_times_colla = total_times_colla.replace(to_replace='stability_strict', value='stability(\u2229)')
total_times_colla = total_times_colla.replace(to_replace='stability_loose', value='stability(\u222A)')
total_times_colla = total_times_colla.replace(to_replace='growth_loose', value='growth(\u222A)')
total_times_colla = total_times_colla.replace(to_replace='shrinkage_loose', value='shrinkage(\u222A)')
total_times_colla = total_times_colla.rename(columns={'time':'time(s)'})

#plot AGG TIME
fig = px.line(total_times_colla, x="interval", y="time_agg", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot SKY TIME
fig = px.line(total_times_colla, x="interval", y="time_sky", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot TOTAL TIME
fig = px.line(total_times_colla, x="interval", y="time_total", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#1b
# skyline exploration for agg by NODE: author.gender,topic AND EDGE: collaborate
total_times = []
event_col = []
total_duration_avg = []
total_duration_agg_avg = []
dominations_dblp_collb = {}
skylines_dblp_collb = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    duration = []
    duration_agg = []
    for j in range(3):
        duration_of_slices = []
        duration_of_slices_agg = []
        for d in range(slices):
            start = time.perf_counter()
            sky,dom,t_agg = SkylinExpl(intvls_sliced[d], attr_val_combs=copy.deepcopy(attr_val_combs_coll),
                          edge_type=['collaboration'], node_type=['author', 'conference'],
                          stc_attrs=['gender', 'topic'], event=i, nodesdf=dataset_sliced['nodes'][d],
                          edgesdf=dataset_sliced['edges'][d], directed=True)
            end = time.perf_counter()
            duration_of_slices.append(end - start)
            duration_of_slices_agg.append(t_agg)
            if d == slices-1:
                dominations_dblp_collb[i] = dom
                skylines_dblp_collb[i] = sky
        duration.append(duration_of_slices)
        duration_agg.append(duration_of_slices_agg)
    duration = pd.DataFrame(duration).T
    duration_avg = list(duration.mean(axis=1))
    duration_agg = pd.DataFrame(duration_agg).T
    duration_agg_avg = list(duration_agg.mean(axis=1))
    total_duration_avg.extend(duration_avg)
    total_duration_agg_avg.extend(duration_agg_avg)
    interval_col = ['up to ' + intvls_sliced[d][-1][-1][0] for d in range(slices)]
    duration['interval'] = interval_col
    total_times.append(duration)
    event_col.extend([i]*slices)

total_times_collb = pd.concat(total_times)
total_times_collb['time_total'] = total_duration_avg
total_times_collb['time_agg'] = total_duration_agg_avg
total_times_collb['time_sky'] = total_times_collb['time_total'] - total_times_collb['time_agg']
total_times_collb['event'] = event_col
total_times_collb = total_times_collb.round(2)
total_times_collb.to_csv('exp_results/dblp/author.gender_topic_edge.coll.csv', header='column_names')
total_times_collb = total_times_collb.replace(to_replace='stability_strict', value='stability(\u2229)')
total_times_collb = total_times_collb.replace(to_replace='stability_loose', value='stability(\u222A)')
total_times_collb = total_times_collb.replace(to_replace='growth_loose', value='growth(\u222A)')
total_times_collb = total_times_collb.replace(to_replace='shrinkage_loose', value='shrinkage(\u222A)')
total_times_collb = total_times_collb.rename(columns={'time':'time(s)'})

# save skyline size
skylines_dblp_collb_size = {key:len(val) for key,val in skylines_dblp_collb.items()}
skylines_dblp_collb_size = pd.Series(skylines_dblp_collb_size)
skylines_dblp_collb_size.name = 'size'
skylines_dblp_collb_size.to_csv('exp_results/dblp/sky_size_author.gender_topic_edge.coll.csv', header='column_names')


# save top-k results
k = 3
top_dblp_collb = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    dom_val = sorted([j[1] for j in dominations_dblp_collb[i].items()])[::-1][:k]
    top={}
    for key,val in dominations_dblp_collb[i].items():
        if val in dom_val:
            key = ast.literal_eval(key)
            tmp = [key[0],key[-1],[key[1][0], key[1][-1]]]
            tmp = str(tmp)
            top[tmp] = [val, sum(key[0][:-1]), i]
    top_dblp_collb[i] = top

top_dblp_collb = [pd.DataFrame.from_dict(top_dblp_collb[i],orient='index') for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']]
top_dblp_collb = pd.concat(top_dblp_collb)
top_dblp_collb.columns = ['dod', 'count sum', 'event']
top_dblp_collb.to_csv('exp_results/dblp/top_author.gender_topic_edge.coll.csv', header='column_names')

# =============================================================================
# #read csv
# total_time_csv = pd.read_csv('exp_results/author.gender_edge.coll.csv', sep=',', index_col=0)
# =============================================================================

#plot AGG TIME
fig = px.line(total_times_collb, x="interval", y="time_agg", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot SKY TIME
fig = px.line(total_times_collb, x="interval", y="time_sky", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot TOTAL TIME
fig = px.line(total_times_collb, x="interval", y="time_total", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()


#2
# skyline exploration for agg by NODE: author.gender, conference.topic AND EDGE: publish
total_times = []
event_col = []
total_duration_avg = []
total_duration_agg_avg = []
dominations_dblp_publ = {}
skylines_dblp_publ = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    duration = []
    duration_agg = []
    for j in range(3):
        duration_of_slices = []
        duration_of_slices_agg = []
        for d in range(slices):
            start = time.perf_counter()
            sky,dom,t_agg = SkylinExpl(intvls_sliced[d], attr_val_combs=copy.deepcopy(attr_val_combs_publ),
                          edge_type=['publication'], node_type=['author', 'conference'],
                          stc_attrs=['gender','topic'], event=i, nodesdf=dataset_sliced['nodes'][d],
                          edgesdf=dataset_sliced['edges'][d], directed=True)
            end = time.perf_counter()
            duration_of_slices.append(end - start)
            duration_of_slices_agg.append(t_agg)
            if d == slices-1:
                dominations_dblp_publ[i] = dom
                skylines_dblp_publ[i] = sky
        duration.append(duration_of_slices)
        duration_agg.append(duration_of_slices_agg)
    duration = pd.DataFrame(duration).T
    duration_avg = list(duration.mean(axis=1))
    duration_agg = pd.DataFrame(duration_agg).T
    duration_agg_avg = list(duration_agg.mean(axis=1))
    total_duration_avg.extend(duration_avg)
    total_duration_agg_avg.extend(duration_agg_avg)
    interval_col = ['up to ' + intvls_sliced[d][-1][-1][0] for d in range(slices)]
    duration['interval'] = interval_col
    total_times.append(duration)
    event_col.extend([i]*slices)


total_times_publ = pd.concat(total_times)
total_times_publ['time_total'] = total_duration_avg
total_times_publ['time_agg'] = total_duration_agg_avg
total_times_publ['time_sky'] = total_times_publ['time_total'] - total_times_publ['time_agg']
total_times_publ['event'] = event_col
total_times_publ = total_times_publ.round(2)
total_times_publ.to_csv('exp_results/dblp/author.gender_topic_edge.publ.csv', header='column_names')
total_times_publ = total_times_publ.replace(to_replace='stability_strict', value='stability(\u2229)')
total_times_publ = total_times_publ.replace(to_replace='stability_loose', value='stability(\u222A)')
total_times_publ = total_times_publ.replace(to_replace='growth_loose', value='growth(\u222A)')
total_times_publ = total_times_publ.replace(to_replace='shrinkage_loose', value='shrinkage(\u222A)')
total_times_publ = total_times_publ.rename(columns={'time':'time(s)'})


# save skyline size
skylines_dblp_publ_size = {key:len(val) for key,val in skylines_dblp_publ.items()}
skylines_dblp_publ_size = pd.Series(skylines_dblp_publ_size)
skylines_dblp_publ_size.name = 'size'
skylines_dblp_publ_size.to_csv('exp_results/dblp/sky_size_author.gender_topic_edge.publ.csv', header='column_names')


# save top-k results
k = 3
top_dblp_publ = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    dom_val = sorted([j[1] for j in dominations_dblp_publ[i].items()])[::-1][:k]
    top={}
    for key,val in dominations_dblp_publ[i].items():
        if val in dom_val:
            key = ast.literal_eval(key)
            tmp = [key[0],key[-1],[key[1][0], key[1][-1]]]
            tmp = str(tmp)
            top[tmp] = [val, sum(key[0][:-1]), i]
    top_dblp_publ[i] = top

top_dblp_publ = [pd.DataFrame.from_dict(top_dblp_publ[i],orient='index') for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']]
top_dblp_publ = pd.concat(top_dblp_publ)
top_dblp_publ.columns = ['dod', 'count sum', 'event']
top_dblp_publ.to_csv('exp_results/dblp/top_author.gender_topic_edge.publ.csv', header='column_names')


#plot AGG TIME
fig = px.line(total_times_publ, x="interval", y="time_agg", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot SKY TIME
fig = px.line(total_times_publ, x="interval", y="time_sky", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot TOTAL TIME
fig = px.line(total_times_publ, x="interval", y="time_total", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()


#3
# skyline exploration for agg by NODE: author.gender,conference.topic AND EDGE: collaborate,publish
total_times = []
event_col = []
total_duration_avg = []
total_duration_agg_avg = []
dominations_dblp_coll_publ = {}
skylines_dblp_coll_publ = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    duration = []
    duration_agg = []
    for j in range(3):
        duration_of_slices = []
        duration_of_slices_agg = []
        for d in range(slices):
            start = time.perf_counter()
            sky,dom,t_agg = SkylinExpl(intvls_sliced[d], attr_val_combs=copy.deepcopy(attr_val_combs_coll_publ),
                          edge_type=['collaboration', 'publication'], node_type=['author', 'conference'],
                          stc_attrs=['gender','topic'], event=i, nodesdf=dataset_sliced['nodes'][d],
                          edgesdf=dataset_sliced['edges'][d], directed=True)
            end = time.perf_counter()
            duration_of_slices.append(end - start)
            duration_of_slices_agg.append(t_agg)
            if d == slices-1:
                dominations_dblp_coll_publ[i] = dom
                skylines_dblp_coll_publ[i] = sky
        duration.append(duration_of_slices)
        duration_agg.append(duration_of_slices_agg)
    duration = pd.DataFrame(duration).T
    duration_avg = list(duration.mean(axis=1))
    duration_agg = pd.DataFrame(duration_agg).T
    duration_agg_avg = list(duration_agg.mean(axis=1))
    total_duration_avg.extend(duration_avg)
    total_duration_agg_avg.extend(duration_agg_avg)
    interval_col = ['up to ' + intvls_sliced[d][-1][-1][0] for d in range(slices)]
    duration['interval'] = interval_col
    total_times.append(duration)
    event_col.extend([i]*slices)

total_times_coll_publ = pd.concat(total_times)
total_times_coll_publ['time_total'] = total_duration_avg
total_times_coll_publ['time_agg'] = total_duration_agg_avg
total_times_coll_publ['time_sky'] = total_times_coll_publ['time_total'] - total_times_coll_publ['time_agg']
total_times_coll_publ['event'] = event_col
total_times_coll_publ = total_times_coll_publ.round(2)
total_times_coll_publ.to_csv('exp_results/dblp/author.gender_topic_edge.coll_publ.csv', header='column_names')
total_times_coll_publ = total_times_coll_publ.replace(to_replace='stability_strict', value='stability(\u2229)')
total_times_coll_publ = total_times_coll_publ.replace(to_replace='stability_loose', value='stability(\u222A)')
total_times_coll_publ = total_times_coll_publ.replace(to_replace='growth_loose', value='growth(\u222A)')
total_times_coll_publ = total_times_coll_publ.replace(to_replace='shrinkage_loose', value='shrinkage(\u222A)')
total_times_coll_publ = total_times_coll_publ.rename(columns={'time':'time(s)'})


# save skyline size
skylines_dblp_coll_publ_size = {key:len(val) for key,val in skylines_dblp_coll_publ.items()}
skylines_dblp_coll_publ_size = pd.Series(skylines_dblp_coll_publ_size)
skylines_dblp_coll_publ_size.name = 'size'
skylines_dblp_coll_publ_size.to_csv('exp_results/dblp/sky_size_author.gender_topic_edge.coll_publ.csv', header='column_names')


# save top-k results
k = 3
top_dblp_coll_publ = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    dom_val = sorted([j[1] for j in dominations_dblp_coll_publ[i].items()])[::-1][:k]
    top={}
    for key,val in dominations_dblp_coll_publ[i].items():
        if val in dom_val:
            key = ast.literal_eval(key)
            tmp = [key[0],key[-1],[key[1][0], key[1][-1]]]
            tmp = str(tmp)
            top[tmp] = [val, sum(key[0][:-1]), i]
    top_dblp_coll_publ[i] = top

top_dblp_coll_publ = [pd.DataFrame.from_dict(top_dblp_coll_publ[i],orient='index') for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']]
top_dblp_coll_publ = pd.concat(top_dblp_coll_publ)
top_dblp_coll_publ.columns = ['dod', 'count sum', 'event']
top_dblp_coll_publ.to_csv('exp_results/dblp/top_author.gender_topic_edge.coll_publ.csv', header='column_names')


#plot AGG TIME
fig = px.line(total_times_coll_publ, x="interval", y="time_agg", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot SKY TIME
fig = px.line(total_times_coll_publ, x="interval", y="time_sky", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot TOTAL TIME
fig = px.line(total_times_coll_publ, x="interval", y="time_total", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()




# =============================================================================
# fig = px.line(total_times_publ, x="interval", y="time(s)", color='event')
# fig.show()
# 
# 
# df = px.data.gapminder().query("continent=='Oceania'")
# fig = px.line(df, x="year", y="lifeExp", color='country')
# fig.show()
# 
# #'\u2192'
# #'\u222A'
# #'\u2229'
# 
# =============================================================================




# Read MOVIELENS

nodes_df = pd.read_csv('../skyline_extension/datasets/movielens_dataset/property_graph/nodes.csv', sep=',', index_col=[0,1])
nodes_df.index.names = ['id', 'type']
edges_df = pd.read_csv('../skyline_extension/datasets/movielens_dataset/property_graph/edges.csv', sep=',', index_col=[0,1,2])
time_invar = pd.read_csv('../skyline_extension/datasets/movielens_dataset/property_graph/time_invar.csv', sep=',', index_col=[0,1], dtype=str)
time_invar.index.names = ['id', 'type']
time_var = pd.read_csv('../skyline_extension/datasets/movielens_dataset/property_graph/time_var.csv', sep=',', index_col=[0,1], dtype=str)
time_var.index.names = ['id', 'type']


c=0
intvls = []
for i in range(1,len(edges_df.columns)+1-c):
    intvls.append([list(edges_df.columns[:i]), list(edges_df.columns[i:i+1])])
    c += 1
intvls = intvls[:-1]

# increasing dataset
nodes_sliced = [nodes_df.iloc[:,:2], nodes_df.iloc[:,:3], nodes_df.iloc[:,:4], nodes_df.iloc[:,:5], nodes_df.iloc[:,:6]]
nodes_sliced = [i.loc[~(i==0).all(axis=1)] for i in nodes_sliced]
edges_sliced = [edges_df.iloc[:,:2], edges_df.iloc[:,:3], edges_df.iloc[:,:4], edges_df.iloc[:,:5], edges_df.iloc[:,:6]]
edges_sliced = [i.loc[~(i==0).all(axis=1)] for i in edges_sliced]
dataset_sliced = {'nodes': nodes_sliced, 'edges': edges_sliced}
intvls_sliced = [intvls[:1], intvls[:2], intvls[:3], intvls[:4], intvls[:5]]

slices = len(intvls_sliced)


attr_val_left = time_invar.gender[time_invar.index.get_level_values('type') == 'user'].unique().tolist()
attr_val_right = time_invar.age[time_invar.index.get_level_values('type') == 'user'].unique().tolist()
attr_val_left_right = list(itertools.product(attr_val_left, attr_val_right))

attr_val_combs_corate_gender = list(itertools.product(attr_val_left, attr_val_left))
attr_val_combs_corate_age = list(itertools.product(attr_val_right, attr_val_right))
attr_val_combs_corate_gender_age = list(itertools.product(attr_val_left_right, attr_val_left_right))
attr_val_combs_corate_gender_age = [i[0]+i[1] for i in attr_val_combs_corate_gender_age]



# MOVIELENS EXPERIMENTS
#1
# skyline exploration for agg by NODE: user.gender AND EDGE: corate
total_times = []
event_col = []
total_duration_avg = []
total_duration_agg_avg = []
dominations_ml_cor_gen = {}
skylines_ml_cor_gen = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    duration = []
    duration_agg = []
    for j in range(3):
        duration_of_slices = []
        duration_of_slices_agg = []
        for d in range(slices):
            start = time.perf_counter()
            sky,dom,t_agg = SkylinExpl(intvls_sliced[d], attr_val_combs=copy.deepcopy(attr_val_combs_corate_gender),
                          edge_type=['corate'], node_type=['user'],
                          stc_attrs=['gender'], event=i, nodesdf=dataset_sliced['nodes'][d],
                          edgesdf=dataset_sliced['edges'][d], directed=True)
            end = time.perf_counter()
            duration_of_slices.append(end - start)
            duration_of_slices_agg.append(t_agg)
            if d == slices-1:
                dominations_ml_cor_gen[i] = dom
                skylines_ml_cor_gen[i] = sky
        duration.append(duration_of_slices)
        duration_agg.append(duration_of_slices_agg)
    duration = pd.DataFrame(duration).T
    duration_avg = list(duration.mean(axis=1))
    duration_agg = pd.DataFrame(duration_agg).T
    duration_agg_avg = list(duration_agg.mean(axis=1))
    total_duration_avg.extend(duration_avg)
    total_duration_agg_avg.extend(duration_agg_avg)
    interval_col = ['up to ' + intvls_sliced[d][-1][-1][0] for d in range(slices)]
    duration['interval'] = interval_col
    total_times.append(duration)
    event_col.extend([i]*slices)

total_times_cor_gen = pd.concat(total_times)
total_times_cor_gen['time_total'] = total_duration_avg
total_times_cor_gen['time_agg'] = total_duration_agg_avg
total_times_cor_gen['time_sky'] = total_times_cor_gen['time_total'] - total_times_cor_gen['time_agg']
total_times_cor_gen['event'] = event_col
total_times_cor_gen = total_times_cor_gen.round(2)
total_times_cor_gen.to_csv('exp_results/ml/user.gender_edge.corate_gen.csv', header='column_names')
total_times_cor_gen = total_times_cor_gen.replace(to_replace='stability_strict', value='stability(\u2229)')
total_times_cor_gen = total_times_cor_gen.replace(to_replace='stability_loose', value='stability(\u222A)')
total_times_cor_gen = total_times_cor_gen.replace(to_replace='growth_loose', value='growth(\u222A)')
total_times_cor_gen = total_times_cor_gen.replace(to_replace='shrinkage_loose', value='shrinkage(\u222A)')


# save skyline size
skylines_ml_cor_gen_size = {key:len(val) for key,val in skylines_ml_cor_gen.items()}
skylines_ml_cor_gen_size = pd.Series(skylines_ml_cor_gen_size)
skylines_ml_cor_gen_size.name = 'size'
skylines_ml_cor_gen_size.to_csv('exp_results/ml/sky_size_user.gender_edge.corate_gen.csv', header='column_names')


# save top-k results
k = 3
top_ml_cor_gen = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    dom_val = sorted([j[1] for j in dominations_ml_cor_gen[i].items()])[::-1][:k]
    top={}
    for key,val in dominations_ml_cor_gen[i].items():
        if val in dom_val:
            key = ast.literal_eval(key)
            tmp = [key[0],key[-1],[key[1][0], key[1][-1]]]
            tmp = str(tmp)
            top[tmp] = [val, sum(key[0][:-1]), i]
    top_ml_cor_gen[i] = top

top_ml_cor_gen = [pd.DataFrame.from_dict(top_ml_cor_gen[i],orient='index') for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']]
top_ml_cor_gen = pd.concat(top_ml_cor_gen)
top_ml_cor_gen.columns = ['dod', 'count sum', 'event']
top_ml_cor_gen.to_csv('exp_results/ml/top_user.gender_edge.corate_gen.csv', header='column_names')


#plot AGG TIME
fig = px.line(total_times_cor_gen, x="interval", y="time_agg", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot SKY TIME
fig = px.line(total_times_cor_gen, x="interval", y="time_sky", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot TOTAL TIME
fig = px.line(total_times_cor_gen, x="interval", y="time_total", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()




#2
# skyline exploration for agg by NODE: user.age AND EDGE: corate
total_times = []
event_col = []
total_duration_avg = []
total_duration_agg_avg = []
dominations_ml_cor_age = {}
skylines_ml_cor_age = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    duration = []
    duration_agg = []
    for j in range(3):
        duration_of_slices = []
        duration_of_slices_agg = []
        for d in range(slices):
            start = time.perf_counter()
            sky,dom,t_agg = SkylinExpl(intvls_sliced[d], attr_val_combs=copy.deepcopy(attr_val_combs_corate_age),
                          edge_type=['corate'], node_type=['user'],
                          stc_attrs=['age'], event=i, nodesdf=dataset_sliced['nodes'][d],
                          edgesdf=dataset_sliced['edges'][d], directed=True)
            end = time.perf_counter()
            duration_of_slices.append(end - start)
            duration_of_slices_agg.append(t_agg)
            if d == slices-1:
                dominations_ml_cor_age[i] = dom
                skylines_ml_cor_age[i] = sky
        duration.append(duration_of_slices)
        duration_agg.append(duration_of_slices_agg)
    duration = pd.DataFrame(duration).T
    duration_avg = list(duration.mean(axis=1))
    duration_agg = pd.DataFrame(duration_agg).T
    duration_agg_avg = list(duration_agg.mean(axis=1))
    total_duration_avg.extend(duration_avg)
    total_duration_agg_avg.extend(duration_agg_avg)
    interval_col = ['up to ' + intvls_sliced[d][-1][-1][0] for d in range(slices)]
    duration['interval'] = interval_col
    total_times.append(duration)
    event_col.extend([i]*slices)

total_times_cor_age = pd.concat(total_times)
total_times_cor_age['time_total'] = total_duration_avg
total_times_cor_age['time_agg'] = total_duration_agg_avg
total_times_cor_age['time_sky'] = total_times_cor_age['time_total'] - total_times_cor_age['time_agg']
total_times_cor_age['event'] = event_col
total_times_cor_age = total_times_cor_age.round(2)
total_times_cor_age.to_csv('exp_results/ml/user.gender_edge.corate_age.csv', header='column_names')
total_times_cor_age = total_times_cor_age.replace(to_replace='stability_strict', value='stability(\u2229)')
total_times_cor_age = total_times_cor_age.replace(to_replace='stability_loose', value='stability(\u222A)')
total_times_cor_age = total_times_cor_age.replace(to_replace='growth_loose', value='growth(\u222A)')
total_times_cor_age = total_times_cor_age.replace(to_replace='shrinkage_loose', value='shrinkage(\u222A)')


# save skyline size
skylines_ml_cor_age_size = {key:len(val) for key,val in skylines_ml_cor_age.items()}
skylines_ml_cor_age_size = pd.Series(skylines_ml_cor_age_size)
skylines_ml_cor_age_size.name = 'size'
skylines_ml_cor_age_size.to_csv('exp_results/ml/sky_size_user.gender_edge.corate_age.csv', header='column_names')


# save top-k results
k = 3
top_ml_cor_age = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    dom_val = sorted([j[1] for j in dominations_ml_cor_age[i].items()])[::-1][:k]
    top={}
    for key,val in dominations_ml_cor_age[i].items():
        if val in dom_val:
            key = ast.literal_eval(key)
            tmp = [key[0],key[-1],[key[1][0], key[1][-1]]]
            tmp = str(tmp)
            top[tmp] = [val, sum(key[0][:-1]), i]
    top_ml_cor_age[i] = top

top_ml_cor_age = [pd.DataFrame.from_dict(top_ml_cor_age[i],orient='index') for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']]
top_ml_cor_age = pd.concat(top_ml_cor_age)
top_ml_cor_age.columns = ['dod', 'count sum', 'event']
top_ml_cor_age.to_csv('exp_results/ml/top_user.gender_edge.corate_age.csv', header='column_names')


#plot AGG TIME
fig = px.line(total_times_cor_age, x="interval", y="time_agg", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot SKY TIME
fig = px.line(total_times_cor_age, x="interval", y="time_sky", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot TOTAL TIME
fig = px.line(total_times_cor_age, x="interval", y="time_total", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()


#3
# skyline exploration for agg by NODE: user.gender,age AND EDGE: corate
total_times = []
event_col = []
total_duration_avg = []
total_duration_agg_avg = []
dominations_ml_cor_gen_age = {}
skylines_ml_cor_gen_age = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    duration = []
    duration_agg = []
    for j in range(3):
        duration_of_slices = []
        duration_of_slices_agg = []
        for d in range(slices):
            start = time.perf_counter()
            sky,dom,t_agg = SkylinExpl(intvls_sliced[d], attr_val_combs=copy.deepcopy(attr_val_combs_corate_gender_age),
                          edge_type=['corate'], node_type=['user'],
                          stc_attrs=['gender','age'], event=i, nodesdf=dataset_sliced['nodes'][d],
                          edgesdf=dataset_sliced['edges'][d], directed=True)
            end = time.perf_counter()
            duration_of_slices.append(end - start)
            duration_of_slices_agg.append(t_agg)
            if d == slices-1:
                dominations_ml_cor_gen_age[i] = dom
                skylines_ml_cor_gen_age[i] = sky
        duration.append(duration_of_slices)
        duration_agg.append(duration_of_slices_agg)
    duration = pd.DataFrame(duration).T
    duration_avg = list(duration.mean(axis=1))
    duration_agg = pd.DataFrame(duration_agg).T
    duration_agg_avg = list(duration_agg.mean(axis=1))
    total_duration_avg.extend(duration_avg)
    total_duration_agg_avg.extend(duration_agg_avg)
    interval_col = ['up to ' + intvls_sliced[d][-1][-1][0] for d in range(slices)]
    duration['interval'] = interval_col
    total_times.append(duration)
    event_col.extend([i]*slices)

total_times_cor_gen_age = pd.concat(total_times)
total_times_cor_gen_age['time_total'] = total_duration_avg
total_times_cor_gen_age['time_agg'] = total_duration_agg_avg
total_times_cor_gen_age['time_sky'] = total_times_cor_gen_age['time_total'] - total_times_cor_gen_age['time_agg']
total_times_cor_gen_age['event'] = event_col
total_times_cor_gen_age = total_times_cor_gen_age.round(2)
total_times_cor_gen_age.to_csv('exp_results/ml/user.gender_edge.corate_gen_age.csv', header='column_names')
total_times_cor_gen_age = total_times_cor_gen_age.replace(to_replace='stability_strict', value='stability(\u2229)')
total_times_cor_gen_age = total_times_cor_gen_age.replace(to_replace='stability_loose', value='stability(\u222A)')
total_times_cor_gen_age = total_times_cor_gen_age.replace(to_replace='growth_loose', value='growth(\u222A)')
total_times_cor_gen_age = total_times_cor_gen_age.replace(to_replace='shrinkage_loose', value='shrinkage(\u222A)')


# save skyline size
skylines_ml_cor_gen_age_size = {key:len(val) for key,val in skylines_ml_cor_gen_age.items()}
skylines_ml_cor_gen_age_size = pd.Series(skylines_ml_cor_gen_age_size)
skylines_ml_cor_gen_age_size.name = 'size'
skylines_ml_cor_gen_age_size.to_csv('exp_results/ml/sky_size_user.gender_edge.corate_gen_age.csv', header='column_names')


# save top-k results
k = 3
top_ml_cor_gen_age = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    dom_val = sorted([j[1] for j in dominations_ml_cor_gen_age[i].items()])[::-1][:k]
    top={}
    for key,val in dominations_ml_cor_gen_age[i].items():
        if val in dom_val:
            key = ast.literal_eval(key)
            tmp = [key[0],key[-1],[key[1][0], key[1][-1]]]
            tmp = str(tmp)
            top[tmp] = [val, sum(key[0][:-1]), i]
    top_ml_cor_gen_age[i] = top

top_ml_cor_gen_age = [pd.DataFrame.from_dict(top_ml_cor_gen_age[i],orient='index') for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']]
top_ml_cor_gen_age = pd.concat(top_ml_cor_gen_age)
top_ml_cor_gen_age.columns = ['dod', 'count sum', 'event']
top_ml_cor_gen_age.to_csv('exp_results/ml/top_user.gender_edge.corate_gen_age.csv', header='column_names')


#plot AGG TIME
fig = px.line(total_times_cor_gen_age, x="interval", y="time_agg", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot SKY TIME
fig = px.line(total_times_cor_gen_age, x="interval", y="time_sky", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot TOTAL TIME
fig = px.line(total_times_cor_gen_age, x="interval", y="time_total", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()




# Read PRIMARY SCHOOL

nodes_df = pd.read_csv('../skyline_extension/datasets/school_dataset/property_graph/nodes_df.csv', sep=' ', index_col=[0,1])
nodes_df.index.names = ['id', 'type']
nodes_df['type'] = 'individual'
nodes_df = nodes_df.droplevel('type').set_index('type', append=True)
edges_df = pd.read_csv('../skyline_extension/datasets/school_dataset/property_graph/edges_df.csv', sep=' ', index_col=[0,1,2])
time_invar = pd.read_csv('../skyline_extension/datasets/school_dataset/property_graph/time_invar.csv', sep=' ', index_col=[0,1], dtype=str)
time_invar.index.names = ['id', 'type']
time_invar = time_invar.rename(columns={'class': 'class_grade'})
time_invar['type'] = 'individual'
time_invar = time_invar.droplevel('type').set_index('type', append=True)

c=0
intvls = []
for i in range(1,len(edges_df.columns)+1-c):
    intvls.append([list(edges_df.columns[:i]), list(edges_df.columns[i:i+1])])
    c += 1
intvls = intvls[:-1]


# increasing dataset
nodes_sliced = [nodes_df.iloc[:,:5], nodes_df.iloc[:,:9], nodes_df.iloc[:,:13], nodes_df.iloc[:,:17]]
nodes_sliced = [i.loc[~(i==0).all(axis=1)] for i in nodes_sliced]
edges_sliced = [edges_df.iloc[:,:5], edges_df.iloc[:,:9], edges_df.iloc[:,:13], edges_df.iloc[:,:17]]
edges_sliced = [i.loc[~(i==0).all(axis=1)] for i in edges_sliced]
dataset_sliced = {'nodes': nodes_sliced, 'edges': edges_sliced}
intvls_sliced = [intvls[:4], intvls[:8], intvls[:12], intvls[:16]]

slices = len(intvls_sliced)


attr_val_left = time_invar.gender[time_invar.index.get_level_values('type') == 'individual'].unique().tolist()
attr_val_left.remove('U')
attr_val_right = time_invar.class_grade[time_invar.index.get_level_values('type') == 'individual'].unique().tolist()
attr_val_left_right = list(itertools.product(attr_val_left, attr_val_right))

attr_val_combs_contact_gender = list(itertools.product(attr_val_left, attr_val_left))
attr_val_combs_contact_class = list(itertools.product(attr_val_right, attr_val_right))
attr_val_combs_contact_gender_class = list(itertools.product(attr_val_left_right, attr_val_left_right))

###################

# undirected
attr_val_combs_contact_gender = [sorted(i) for i in attr_val_combs_contact_gender]
attr_val_combs_contact_gender = list(set([tuple(i) for i in attr_val_combs_contact_gender]))

attr_val_combs_contact_class = [sorted(i) for i in attr_val_combs_contact_class]
attr_val_combs_contact_class = list(set([tuple(i) for i in attr_val_combs_contact_class]))

attr_val_combs_contact_gender_class = [sorted(i) for i in attr_val_combs_contact_gender_class]
attr_val_combs_contact_gender_class = list(set([tuple(i) for i in attr_val_combs_contact_gender_class]))

##################

attr_val_combs_contact_gender_class = [i[0]+i[1] for i in attr_val_combs_contact_gender_class]



# PRIMARY SCHOOL EXPERIMENTS

#1
# skyline exploration for agg by NODE: user.gender AND EDGE: contact
total_times = []
event_col = []
total_duration_avg = []
total_duration_agg_avg = []
dominations_ps_con_gen = {}
skylines_ps_con_gen = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    duration = []
    duration_agg = []
    for j in range(3):
        duration_of_slices = []
        duration_of_slices_agg = []
        for d in range(slices):
            start = time.perf_counter()
            sky,dom,t_agg = SkylinExpl(intvls_sliced[d], attr_val_combs=copy.deepcopy(attr_val_combs_contact_gender),
                          edge_type=['contact'], node_type=['individual'],
                          stc_attrs=['gender'], event=i, nodesdf=dataset_sliced['nodes'][d],
                          edgesdf=dataset_sliced['edges'][d], directed=False)
            end = time.perf_counter()
            duration_of_slices.append(end - start)
            duration_of_slices_agg.append(t_agg)
            if d == slices-1:
                dominations_ps_con_gen[i] = dom
                skylines_ps_con_gen[i] = sky
        duration.append(duration_of_slices)
        duration_agg.append(duration_of_slices_agg)
    duration = pd.DataFrame(duration).T
    duration_avg = list(duration.mean(axis=1))
    duration_agg = pd.DataFrame(duration_agg).T
    duration_agg_avg = list(duration_agg.mean(axis=1))
    total_duration_avg.extend(duration_avg)
    total_duration_agg_avg.extend(duration_agg_avg)
    interval_col = ['up to ' + intvls_sliced[d][-1][-1][0] for d in range(slices)]
    duration['interval'] = interval_col
    total_times.append(duration)
    event_col.extend([i]*slices)

total_times_con_gen = pd.concat(total_times)
total_times_con_gen['time_total'] = total_duration_avg
total_times_con_gen['time_agg'] = total_duration_agg_avg
total_times_con_gen['time_sky'] = total_times_con_gen['time_total'] - total_times_con_gen['time_agg']
total_times_con_gen['event'] = event_col
total_times_con_gen = total_times_con_gen.round(2)
total_times_con_gen.to_csv('exp_results/ps/user.gender_edge.contact.csv', header='column_names')
total_times_con_gen = total_times_con_gen.replace(to_replace='stability_strict', value='stability(\u2229)')
total_times_con_gen = total_times_con_gen.replace(to_replace='stability_loose', value='stability(\u222A)')
total_times_con_gen = total_times_con_gen.replace(to_replace='growth_loose', value='growth(\u222A)')
total_times_con_gen = total_times_con_gen.replace(to_replace='shrinkage_loose', value='shrinkage(\u222A)')


# save skyline size
skylines_ps_con_gen_size = {key:len(val) for key,val in skylines_ps_con_gen.items()}
skylines_ps_con_gen_size = pd.Series(skylines_ps_con_gen_size)
skylines_ps_con_gen_size.name = 'size'
skylines_ps_con_gen_size.to_csv('exp_results/ps/sky_size_user.gender_edge.contact.csv', header='column_names')


# save top-k results
k = 3
top_ps_con_gen = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    dom_val = sorted([j[1] for j in dominations_ps_con_gen[i].items()])[::-1][:k]
    top={}
    for key,val in dominations_ps_con_gen[i].items():
        if val in dom_val:
            key = ast.literal_eval(key)
            tmp = [key[0],key[-1],[key[1][0], key[1][-1]]]
            tmp = str(tmp)
            top[tmp] = [val, sum(key[0][:-1]), i]
    top_ps_con_gen[i] = top

top_ps_con_gen = [pd.DataFrame.from_dict(top_ps_con_gen[i],orient='index') for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']]
top_ps_con_gen = pd.concat(top_ps_con_gen)
top_ps_con_gen.columns = ['dod', 'count sum', 'event']
top_ps_con_gen.to_csv('exp_results/ps/top_user.gender_edge.contact.csv', header='column_names')


#plot AGG TIME
fig = px.line(total_times_con_gen, x="interval", y="time_agg", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot SKY TIME
fig = px.line(total_times_con_gen, x="interval", y="time_sky", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot TOTAL TIME
fig = px.line(total_times_con_gen, x="interval", y="time_total", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()



#2
# skyline exploration for agg by NODE: user.class AND EDGE: contact
total_times = []
event_col = []
total_duration_avg = []
total_duration_agg_avg = []
dominations_ps_con_class = {}
skylines_ps_con_class = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    duration = []
    duration_agg = []
    for j in range(3):
        duration_of_slices = []
        duration_of_slices_agg = []
        for d in range(slices):
            start = time.perf_counter()
            sky,dom,t_agg = SkylinExpl(intvls_sliced[d], attr_val_combs=copy.deepcopy(attr_val_combs_contact_class),
                          edge_type=['contact'], node_type=['individual'],
                          stc_attrs=['class_grade'], event=i, nodesdf=dataset_sliced['nodes'][d],
                          edgesdf=dataset_sliced['edges'][d], directed=False)
            end = time.perf_counter()
            duration_of_slices.append(end - start)
            duration_of_slices_agg.append(t_agg)
            if d == slices-1:
                dominations_ps_con_class[i] = dom
                skylines_ps_con_class[i] = sky
        duration.append(duration_of_slices)
        duration_agg.append(duration_of_slices_agg)
    duration = pd.DataFrame(duration).T
    duration_avg = list(duration.mean(axis=1))
    duration_agg = pd.DataFrame(duration_agg).T
    duration_agg_avg = list(duration_agg.mean(axis=1))
    total_duration_avg.extend(duration_avg)
    total_duration_agg_avg.extend(duration_agg_avg)
    interval_col = ['up to ' + intvls_sliced[d][-1][-1][0] for d in range(slices)]
    duration['interval'] = interval_col
    total_times.append(duration)
    event_col.extend([i]*slices)

total_times_con_class = pd.concat(total_times)
total_times_con_class['time_total'] = total_duration_avg
total_times_con_class['time_agg'] = total_duration_agg_avg
total_times_con_class['time_sky'] = total_times_con_class['time_total'] - total_times_con_class['time_agg']
total_times_con_class['event'] = event_col
total_times_con_class = total_times_con_class.round(2)
total_times_con_class.to_csv('exp_results/ps/user.class_edge.contact.csv', header='column_names')
total_times_con_class = total_times_con_class.replace(to_replace='stability_strict', value='stability(\u2229)')
total_times_con_class = total_times_con_class.replace(to_replace='stability_loose', value='stability(\u222A)')
total_times_con_class = total_times_con_class.replace(to_replace='growth_loose', value='growth(\u222A)')
total_times_con_class = total_times_con_class.replace(to_replace='shrinkage_loose', value='shrinkage(\u222A)')


# save skyline size
skylines_ps_con_class_size = {key:len(val) for key,val in skylines_ps_con_class.items()}
skylines_ps_con_class_size = pd.Series(skylines_ps_con_class_size)
skylines_ps_con_class_size.name = 'size'
skylines_ps_con_class_size.to_csv('exp_results/ps/sky_size_user.class_edge.contact.csv', header='column_names')


# save top-k results
k = 3
top_ps_con_class = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    dom_val = sorted([j[1] for j in dominations_ps_con_class[i].items()])[::-1][:k]
    top={}
    for key,val in dominations_ps_con_class[i].items():
        if val in dom_val:
            key = ast.literal_eval(key)
            tmp = [key[0],key[-1],[key[1][0], key[1][-1]]]
            tmp = str(tmp)
            top[tmp] = [val, sum(key[0][:-1]), i]
    top_ps_con_class[i] = top

top_ps_con_class = [pd.DataFrame.from_dict(top_ps_con_class[i],orient='index') for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']]
top_ps_con_class = pd.concat(top_ps_con_class)
top_ps_con_class.columns = ['dod', 'count sum', 'event']
top_ps_con_class.to_csv('exp_results/ps/top_user.class_edge.contact.csv', header='column_names')


#plot AGG TIME
fig = px.line(total_times_con_class, x="interval", y="time_agg", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot SKY TIME
fig = px.line(total_times_con_class, x="interval", y="time_sky", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot TOTAL TIME
fig = px.line(total_times_con_class, x="interval", y="time_total", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()


#3
# skyline exploration for agg by NODE: user.gender,class AND EDGE: contact
total_times = []
event_col = []
total_duration_avg = []
total_duration_agg_avg = []
dominations_ps_con_gen_class = {}
skylines_ps_con_gen_class = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    duration = []
    duration_agg = []
    for j in range(3):
        duration_of_slices = []
        duration_of_slices_agg = []
        for d in range(slices):
            start = time.perf_counter()
            sky,dom,t_agg = SkylinExpl(intvls_sliced[d], attr_val_combs=copy.deepcopy(attr_val_combs_contact_gender_class),
                          edge_type=['contact'], node_type=['individual'],
                          stc_attrs=['gender','class_grade'], event=i, nodesdf=dataset_sliced['nodes'][d],
                          edgesdf=dataset_sliced['edges'][d], directed=False)
            end = time.perf_counter()
            duration_of_slices.append(end - start)
            duration_of_slices_agg.append(t_agg)
            if d == slices-1:
                dominations_ps_con_gen_class[i] = dom
                skylines_ps_con_gen_class[i] = sky
        duration.append(duration_of_slices)
        duration_agg.append(duration_of_slices_agg)
    duration = pd.DataFrame(duration).T
    duration_avg = list(duration.mean(axis=1))
    duration_agg = pd.DataFrame(duration_agg).T
    duration_agg_avg = list(duration_agg.mean(axis=1))
    total_duration_avg.extend(duration_avg)
    total_duration_agg_avg.extend(duration_agg_avg)
    interval_col = ['up to ' + intvls_sliced[d][-1][-1][0] for d in range(slices)]
    duration['interval'] = interval_col
    total_times.append(duration)
    event_col.extend([i]*slices)

total_times_con_gen_class = pd.concat(total_times)
total_times_con_gen_class['time_total'] = total_duration_avg
total_times_con_gen_class['time_agg'] = total_duration_agg_avg
total_times_con_gen_class['time_sky'] = total_times_con_gen_class['time_total'] - total_times_con_gen_class['time_agg']
total_times_con_gen_class['event'] = event_col
total_times_con_gen_class = total_times_con_gen_class.round(2)
total_times_con_gen_class.to_csv('exp_results/ps/user.gen_class_edge.contact.csv', header='column_names')
total_times_con_gen_class = total_times_con_gen_class.replace(to_replace='stability_strict', value='stability(\u2229)')
total_times_con_gen_class = total_times_con_gen_class.replace(to_replace='stability_loose', value='stability(\u222A)')
total_times_con_gen_class = total_times_con_gen_class.replace(to_replace='growth_loose', value='growth(\u222A)')
total_times_con_gen_class = total_times_con_gen_class.replace(to_replace='shrinkage_loose', value='shrinkage(\u222A)')


# save skyline size
skylines_ps_con_gen_class_size = {key:len(val) for key,val in skylines_ps_con_gen_class.items()}
skylines_ps_con_gen_class_size = pd.Series(skylines_ps_con_gen_class_size)
skylines_ps_con_gen_class_size.name = 'size'
skylines_ps_con_gen_class_size.to_csv('exp_results/ps/sky_size_user.gen_class_edge.contact.csv', header='column_names')


# save top-k results
k = 3
top_ps_con_gen_class = {}
for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']:
    dom_val = sorted([j[1] for j in dominations_ps_con_gen_class[i].items()])[::-1][:k]
    top={}
    for key,val in dominations_ps_con_gen_class[i].items():
        if val in dom_val:
            key = ast.literal_eval(key)
            tmp = [key[0],key[-1],[key[1][0], key[1][-1]]]
            tmp = str(tmp)
            top[tmp] = [val, sum(key[0][:-1]), i]
    top_ps_con_gen_class[i] = top

top_ps_con_gen_class = [pd.DataFrame.from_dict(top_ps_con_gen_class[i],orient='index') for i in ['stability_strict', 'stability_loose', 'growth_loose', 'shrinkage_loose']]
top_ps_con_gen_class = pd.concat(top_ps_con_gen_class)
top_ps_con_gen_class.columns = ['dod', 'count sum', 'event']
top_ps_con_gen_class.to_csv('exp_results/ps/top_user.gen_class_edge.contact.csv', header='column_names')


#plot AGG TIME
fig = px.line(total_times_con_gen_class, x="interval", y="time_agg", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot SKY TIME
fig = px.line(total_times_con_gen_class, x="interval", y="time_sky", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()

#plot TOTAL TIME
fig = px.line(total_times_con_gen_class, x="interval", y="time_total", color='event')
fig.update_layout(font=dict(size=26),
                  legend=dict(x=0.05,
                              y=1,
                              bgcolor = 'rgba(0,0,0,0)'),
                  legend_title=None,
                  yaxis_title="time(s)",)
fig.show()



#########################################################


# RADAR CHARTS
# radar chart

#colors = ['#F79494', '#94F7AC', '#949BF7']
colors = ['#F79494', '#94F7AC', '#a6aef5']


# PS contact.class / stab_strict
import plotly.graph_objects as go



df = pd.DataFrame([[0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 9],
                   [0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 10],
                   [0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 11],
                 ])


df.columns = attr_val_combs_contact_class+['length']

df = df.loc[:, (df != 0).any(axis=0)]

#normalize
df = df/df.max()

attr = df.columns.tolist()
attr = [str(i) for i in attr]
attr.append(attr[0])


fig = go.Figure()

fig.add_trace(go.Scatterpolar(marker_color=colors[0],
      r=df.iloc[0,:].values.tolist()+[df.iloc[0,0]],
      theta=attr,
      fill='toself',
      name='Skyline A'
))
fig.add_trace(go.Scatterpolar(marker_color=colors[1],
      r=df.iloc[1,:].values.tolist()+[df.iloc[1,0]],
      theta=attr,
      fill='toself',
      name='Skyline B'
))
fig.add_trace(go.Scatterpolar(marker_color=colors[2],
      r=df.iloc[2,:].values.tolist()+[df.iloc[2,0]],
      theta=attr,
      fill='toself',
      name='Skyline C'
))

fig.update_layout(font=dict(size=17),
  polar=dict(
    radialaxis=dict(
      visible=True,
      #range=[0, 5]
    )),
  showlegend=False
)

fig.show()


# PS contact.class / stab_strict
# top-3 based on the length

df = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 16],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 15],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 15],
                 ])


df.columns = attr_val_combs_contact_class+['length']

df = df.loc[:, (df != 0).any(axis=0)]

#normalize
df = df/df.max()

attr = df.columns.tolist()
attr = [str(i) for i in attr]
attr.append(attr[0])


fig = go.Figure()

fig.add_trace(go.Scatterpolar(marker_color=colors[2],
      r=df.iloc[2,:].values.tolist()+[df.iloc[2,0]],
      theta=attr,
      fill='toself',
      name='Skyline C'
))

fig.add_trace(go.Scatterpolar(marker_color=colors[1],
      r=df.iloc[1,:].values.tolist()+[df.iloc[1,0]],
      theta=attr,
      fill='toself',
      name='Skyline B'
))

fig.add_trace(go.Scatterpolar(marker_color=colors[0],
      r=df.iloc[0,:].values.tolist()+[df.iloc[0,0]],
      theta=attr,
      fill='toself',
      name='Skyline A'
))



fig.update_layout(font=dict(size=17),
  polar=dict(
    radialaxis=dict(
      visible=True,
      #range=[0, 5]
    )),
  showlegend=False
)

fig.show()




# radar chart
# PS contact.class / growth loose

import plotly.graph_objects as go


df_init = pd.DataFrame([[0, 11, 0, 2, 0, 0, 1, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 43, 0, 0, 20, 0, 0, 0, 0, 56, 29, 0, 0, 0, 0, 1, 0, 11, 0, 76, 0, 0, 58, 3, 3, 21, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 10, 0, 1, 16],
                   [0, 27, 0, 0, 0, 0, 0, 0, 0, 8, 0, 13, 5, 0, 11, 0, 2, 0, 18, 2, 4, 16, 1, 0, 0, 2, 0, 0, 0, 32, 5, 0, 0, 0, 0, 0, 0, 7, 40, 0, 5, 0, 0, 0, 0, 0, 19, 3, 3, 0, 3, 32, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 31, 0, 0, 0, 13],
                   [0, 0, 0, 1, 0, 0, 0, 2, 2, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 3, 3, 19, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 9],
                 ])#new


df = pd.DataFrame(np.where(df_init.iloc[:, :-1] <= 10, 0, df_init.iloc[:, :-1]))

df['length'] = df_init.iloc[:,-1]

df.columns = attr_val_combs_contact_class+['length']

df = df.loc[:, (df != 0).any(axis=0)]

#normalize
df = df/df.max()

attr = df.columns.tolist()
attr = [str(i) for i in attr]
attr.append(attr[0])


fig = go.Figure()

fig.add_trace(go.Scatterpolar(marker_color=colors[0],
      r=df.iloc[0,:].values.tolist()+[df.iloc[0,0]],
      theta=attr,
      fill='toself',
      name='Skyline A'
))
fig.add_trace(go.Scatterpolar(marker_color=colors[1],
      r=df.iloc[1,:].values.tolist()+[df.iloc[1,0]],
      theta=attr,
      fill='toself',
      name='Skyline B'
))
fig.add_trace(go.Scatterpolar(marker_color=colors[2],
      r=df.iloc[2,:].values.tolist()+[df.iloc[2,0]],
      theta=attr,
      fill='toself',
      name='Skyline C'
))

fig.update_layout(font=dict(size=17),
  polar=dict(
    radialaxis=dict(
      visible=True,
      #range=[0, 5]
    )),
  showlegend=False
)

fig.show()




########################################################################

# publ dblp
#top-3 stability strict
df = pd.DataFrame([[3, 0, 23, 1, 3, 1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10],
                   [5, 0, 25, 1, 4, 1, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9],
                   [6, 7, 31, 2, 5, 1, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 6, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 8]
                   ])


df.columns = attr_val_combs_publ+['length']

df = df.loc[:, (df != 0).any(axis=0)]

#normalize
df = df/df.max()

attr = df.columns.tolist()
attr = [str(i) for i in attr]
attr = [ast.literal_eval(tpl) for tpl in attr[:-1]]
attr = [str(('F', tpl[1])) if tpl[0]=='female' else tpl for tpl in attr]
attr = [str(('M', tpl[1])) if tpl[0]=='male' else tpl for tpl in attr]
attr.append('length')
attr.append(attr[0])


import plotly.graph_objects as go


fig = go.Figure()

fig.add_trace(go.Scatterpolar(marker_color=colors[2],
      r=df.iloc[2,:].values.tolist()+[df.iloc[2,0]],
      theta=attr,
      fill='toself',
      name='Skyline C'
))

fig.add_trace(go.Scatterpolar(marker_color=colors[1],
      r=df.iloc[1,:].values.tolist()+[df.iloc[1,0]],
      theta=attr,
      fill='toself',
      name='Skyline B',
      #line={'width': 7}
))

fig.add_trace(go.Scatterpolar(marker_color=colors[0],
      r=df.iloc[0,:].values.tolist()+[df.iloc[0,0]],
      theta=attr,
      fill='toself',
      name='Skyline A'
))

fig.update_layout(font=dict(size=15),
  polar=dict(
    radialaxis=dict(
      visible=True,
      #range=[0, 5]
    )),
  showlegend=False
)

fig.show()


# publ dblp
#top-3 stability strict
# top-3 based on the length

df = pd.DataFrame([[0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16],
                   [0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16],
                   [0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15],
                 ])


df.columns = attr_val_combs_publ+['length']

df = df.loc[:, (df != 0).any(axis=0)]

#normalize
df = df/df.max()

attr = df.columns.tolist()
attr = [str(i) for i in attr]
attr = [ast.literal_eval(tpl) for tpl in attr[:-1]]
attr = [str(('F', tpl[1])) if tpl[0]=='female' else tpl for tpl in attr]
attr = [str(('M', tpl[1])) if tpl[0]=='male' else tpl for tpl in attr]
attr.append('length')
attr.append(attr[0])


fig = go.Figure()




fig.add_trace(go.Scatterpolar(marker_color=colors[2],
      r=df.iloc[2,:].values.tolist()+[df.iloc[2,0]],
      theta=attr,
      fill='toself',
      name='Skyline C'
))



fig.add_trace(go.Scatterpolar(marker_color=colors[0],
      r=df.iloc[0,:].values.tolist()+[df.iloc[0,0]],
      theta=attr,
      fill='toself',
      name='Skyline A'
))

fig.add_trace(go.Scatterpolar(marker_color=colors[1],
      r=df.iloc[1,:].values.tolist()+[df.iloc[1,0]],
      theta=attr,
      fill='toself',
      name='Skyline B'
))

fig.update_layout(font=dict(size=17),
  polar=dict(
    radialaxis=dict(
      visible=True,
      #range=[0, 5]
    )),
  showlegend=False
)

fig.show()



# publ dblp
#top-3 growth loose
df_init = pd.DataFrame([[1181, 1371, 688, 531, 300, 193, 217, 606, 357, 0, 0, 26, 0, 8, 0, 0, 175, 239, 99, 62, 76, 33, 43, 104, 62, 0, 0, 7, 0, 1, 0, 0, 16],
                   [1567, 2038, 986, 946, 304, 278, 248, 631, 595, 0, 0, 6, 0, 20, 35, 0, 255, 317, 145, 144, 58, 43, 35, 124, 157, 0, 0, 3, 0, 5, 1, 0, 5],
                   [1603, 2030, 1262, 1038, 361, 224, 226, 586, 543, 0, 0, 3, 0, 4, 70, 27, 260, 297, 177, 148, 75, 36, 68, 94, 147, 0, 0, 0, 0, 1, 3, 15, 19]
                   ])

df = pd.DataFrame(np.where(df_init.iloc[:, :-1] <= 50, 0, df_init.iloc[:, :-1]))


df['length'] = df_init.iloc[:,-1]

df.columns = attr_val_combs_publ+['length']

df = df.loc[:, (df != 0).any(axis=0)]

#normalize
df = df/df.max()

attr = df.columns.tolist()
attr = [str(i) for i in attr]
attr = [ast.literal_eval(tpl) for tpl in attr[:-1]]
attr = [str(('F', tpl[1])) if tpl[0]=='female' else tpl for tpl in attr]
attr = [str(('M', tpl[1])) if tpl[0]=='male' else tpl for tpl in attr]
attr.append('length')
attr.append(attr[0])

import plotly.graph_objects as go


fig = go.Figure()

fig.add_trace(go.Scatterpolar(marker_color=colors[2],
      r=df.iloc[2,:].values.tolist()+[df.iloc[2,0]],
      theta=attr,
      fill='toself',
      name='Skyline C'
))


fig.add_trace(go.Scatterpolar(marker_color=colors[1],
      r=df.iloc[1,:].values.tolist()+[df.iloc[1,0]],
      theta=attr,
      fill='toself',
      name='Skyline B'
))

fig.add_trace(go.Scatterpolar(marker_color=colors[0],
      r=df.iloc[0,:].values.tolist()+[df.iloc[0,0]],
      theta=attr,
      fill='toself',
      name='Skyline A'
))


fig.update_layout(font=dict(size=15),
  polar=dict(
    radialaxis=dict(
      visible=True,
      #range=[0, 5]
    )),
  showlegend=False
)

fig.show()



