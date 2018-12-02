#Example Call:
#for n in range(3):
#    score.append(weighted_overlap(lambda s,t: pw.baseline(s,t,node_dict,node_info)[n],random.sample(true_edges,500),random.sample(no_citation,500)))

def weighted_overlap(f,true_pairs,false_pairs):
    true_f_values = [f(p[0],p[1]) for p in true_pairs]
    true_mean = np.mean(true_f_values)
    false_f_values = [f(p[0],p[1]) for p in false_pairs]
    false_mean = np.mean(false_f_values)
    t_true = len(true_f_values)
    t_false = len(false_f_values)
    
    if true_mean>=false_mean:
        m = max(false_f_values)
        #could be improved by iteratively adding instead of redoing len(list) for every val
        f_overlaps = [len([f for f in false_f_values if f>=val]) for val in true_f_values if val<=m]
        f_overlaps = sum(f_overlaps) / ( t_true*t_false )
    else:
        m = min(false_f_values)
        #could be improved by iteratively adding instead of redoing len(list) for every val
        f_overlaps = [len([f for f in false_f_values if f<=val]) for val in true_f_values if val>=m]
        f_overlaps = sum(f_overlaps) / ( t_true*t_false )
    
    return f_overlaps

