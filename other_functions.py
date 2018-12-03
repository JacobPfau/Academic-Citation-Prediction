from sklearn.linear_model import LogisticRegression as lr
import numpy as np

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
    

#Example Call:
# errors=[]
# for n in range(4):
#     errors.append(error_overlap(lambda s,t: pw.Max_Sim(s,t,l,graph,node_dict)[3],lambda s,t: pw.succ_pred(
#         s,t,graph)[n],random.sample(training_set,2000)))

def error_overlap(f,g,sample_set):
    test_set = np.array(sample_set[:int(len(sample_set)/2)])
    train_set = np.array(sample_set[int(len(sample_set)/2):])
    modelf = lr(penalty='l1',solver='liblinear').fit(np.array([f(p[0],p[1]) for p in train_set]).reshape(-1, 1)
                                                     ,train_set[:,2])
    modelg = lr(penalty='l1',solver='liblinear').fit(np.array([g(p[0],p[1]) for p in train_set]).reshape(-1, 1)
                                                     ,train_set[:,2])
    predsf = modelf.predict(np.array([f(p[0],p[1]) for p in test_set]).reshape(-1, 1))
    predsg = modelg.predict(np.array([g(p[0],p[1]) for p in test_set]).reshape(-1, 1))
    
    first = False
    counts=[0,0,0]
    for i in range(len(predsf)):
        if predsf[i]!=test_set[:,2][i]: 
            first = True
            counts[0]+=1
        if predsg[i]!=test_set[:,2][i]: 
            counts[1]+=1
            if first==True: counts[2]+=1
        first = False
        
    return counts[2]/min([counts[0],counts[1]])