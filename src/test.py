import pickle
import sys
with open('../DATA/ordered140.pkl', 'rb') as f:
  length, labels = pickle.load(f)

acc_labels = []
tmp = labels[:]
for i in range(len(length)-1): 
  try:
    acc_labels.append(tmp[:length[i]])
    tmp = tmp[length[i]:]
  except:
    print(i)
    break;

acc_labels.append(tmp)
with open('labels.pkl', 'wb') as f:
  pickle.dump(acc_labels, f)
print(length[0:15])
print(len(acc_labels[0]))
print(len(acc_labels[1]))
