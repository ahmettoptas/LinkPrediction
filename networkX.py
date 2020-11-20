import networkx as nx
import pandas as pd

data = pd.read_csv('ca-cit-HepPh.txt', header=None).drop(0)
data.head()
data = data[0].str.split(" ", expand=True)

data.drop(columns=2, inplace=True)

data.rename(columns={0: "first node", 1: "second node", 3: "timestamp"}, inplace=True)
data["timestamp"] = data["timestamp"].astype(int)
data["first node"] = data["first node"].astype(int)
data["second node"] = data["second node"].astype(int)

data = data.sort_values(by=["timestamp"])
print(data.head())
print(data.tail())
print()

G3 = nx.from_pandas_edgelist(data, "first node", "second node", "timestamp")

size = G3.number_of_edges()
print(size)

allEdgesData = nx.to_pandas_edgelist(G3)
allEdgesData = allEdgesData.sort_values(by=["timestamp"])
allEdgesData["label"] = 1

def size2_funct(values_dict):
    temp = []
    counter = 0
    for i in values_dict:
        if len(i) == 3:
            temp.append(i[2])
            counter += 1
        if counter == 150:
            break
    return temp


def size3_funct(values_dict):
    temp = []
    counter1 = 0
    for values in values_dict:
        temp_len = len(values)

        if temp_len < 4:
            continue

        if counter1 < 25 and temp_len == 4:
            temp.append(values[3])
            counter1 += 1
        else:
            break
    return temp

f1 = open("deneme.txt","r+")
f2 = open("deneme34.txt", "r+")     #this file opening is to read negative linked file which is written by us.

#this for loop is to write negative links to file.
"""for i, j in paths:
    dict[i] = size3_funct(j.values())
    f.write(str(i) + "\t" + dict[i].__str__() + "\n")"""

label0graph = nx.Graph()

#I write and read 2 files actually. One is negative links which are 2 hops away each other. Other one is 3 steps away each other.
for line in f1:
    counter = 0
    splits = line.split("\t")
    node = int(splits[0])
    list1 = splits[1]

    if list1.count(","):
        res = list1[:-1].strip('][').split(', ')
        for i in res:
            label0graph.add_edge(node, int(i))
            counter += 1
            if counter == 85:
                break
    else:
        res = list1[:-1].strip('][')
        if len(res) > 0:
            label0graph.add_edge(node, int(res))

for line in f2:
    splits = line.split("\t")
    node = int(splits[0])
    list1 = splits[1]

    if list1.count(","):
        res = list1[:-1].strip('][').split(', ')
        for i in res:
            label0graph.add_edge(node, int(i))
    else:
        res = list1[:-1].strip('][')
        if len(res) > 0:
            label0graph.add_edge(node, int(res))

print(label0graph.number_of_edges(), label0graph.number_of_nodes()) #check negative link size

data_before10 = allEdgesData.loc[allEdgesData.timestamp < 1015887601]
data_after10 = allEdgesData.loc[allEdgesData.timestamp >= 1015887601]

data_sample = data_after10.sample(n=1000000, random_state=1)    #to add to data_before10 because of there is almost 3 times amount data comparing to data_before10
rest_part_sec = data_after10.drop(data_sample.index)        #rest of data_after10

data_firstHalf = data_before10.append(data_sample)
# print(data_firstHalf.sample(10))
earlyGraph = nx.from_pandas_edgelist(data_firstHalf, "source", "target", ["timestamp", "label"])

print(data_firstHalf.sample(n=5))
print()
print(rest_part_sec.sample(n=5))

label0graphData = nx.to_pandas_edgelist(label0graph)
label0graphData["label"] = 0
print("label0",label0graphData.sample(5))

Label0FirstHalf = label0graphData.sample(frac=0.5, random_state=1)
Label0SecHalf = label0graphData.drop(Label0FirstHalf.index)

mixedDataFirstHalf = data_firstHalf.append(Label0FirstHalf)
mixedDataSecHalf = rest_part_sec.append(Label0SecHalf)

print("mixed data first half:\n",mixedDataFirstHalf.sample(20))
print()
print("mixed data second half:\n",mixedDataSecHalf.sample(20))




#Link Prediction algorithms for pandas from networkX library

#jaccard_coefficient
def jaccard(graph, node1, node2):
    pred = nx.jaccard_coefficient(graph, [(node1, node2)])
    try:
        for u, v, p in pred:
            return p
    except:
        return 0.00

#adamic_adar
def adamic(graph, node1, node2):
    pred = nx.adamic_adar_index(graph, [(node1, node2)])
    try:
        for u, v, p in pred:
            return p
    except:
        return 0.00

#preferential_attachment
def pref_at(graph, node1, node2):
    pred = nx.preferential_attachment(graph, [(node1, node2)])
    try:
        for u, v, p in pred:
            return p
    except:
        return 0.00

#common_neighbors
def common_n(graph, node1, node2):
    try:
        pred = nx.common_neighbors(graph, node1, node2)
        common_number = len(list(pred))
        return common_number
    except:
        return 0.00


mixedDataFirstHalf['jaccard'] = mixedDataFirstHalf.apply(lambda x: jaccard(earlyGraph, x["source"], x["target"]),
                                                         axis=1)
mixedDataSecHalf['jaccard'] = mixedDataSecHalf.apply(lambda x: jaccard(earlyGraph, x["source"], x["target"]), axis=1)
#mixedDataFirstHalf['jaccard'].to_csv("mixedDataFirstHalf['jaccard'].csv")
#mixedDataSecHalf['jaccard'].to_csv("mixedDataSecHalf['jaccard'].csv")

mixedDataFirstHalf['adamic'] = mixedDataFirstHalf.apply(lambda x: adamic(earlyGraph, x["source"], x["target"]), axis=1)
mixedDataSecHalf['adamic'] = mixedDataSecHalf.apply(lambda x: adamic(earlyGraph, x["source"], x["target"]), axis=1)
#mixedDataFirstHalf['adamic'].to_csv("mixedDataFirstHalf['adamic'].csv")
#mixedDataSecHalf['adamic'].to_csv("mixedDataSecHalf['adamic'].csv")

mixedDataFirstHalf['pref_at'] = mixedDataFirstHalf.apply(lambda x: pref_at(earlyGraph, x["source"], x["target"]),
                                                         axis=1)
mixedDataSecHalf['pref_at'] = mixedDataSecHalf.apply(lambda x: pref_at(earlyGraph, x["source"], x["target"]), axis=1)
#mixedDataFirstHalf['pref_at'].to_csv("mixedDataFirstHalf['pref_at'].csv")
#mixedDataSecHalf['pref_at'].to_csv("mixedDataSecHalf['pref_at'].csv")

mixedDataFirstHalf['common_n'] = mixedDataFirstHalf.apply(lambda x: common_n(earlyGraph, x["source"], x["target"]),
                                                          axis=1)
mixedDataSecHalf['common_n'] = mixedDataSecHalf.apply(lambda x: common_n(earlyGraph, x["source"], x["target"]), axis=1)
#mixedDataFirstHalf['common_n'].to_csv("mixedDataFirstHalf['common_n'].csv")
#mixedDataSecHalf['common_n'].to_csv("mixedDataSecHalf['common_n'].csv")

print(mixedDataFirstHalf.describe(include="all"))
mixedDataFirstHalf.to_csv("mixedDataFirstHalf.csv")
mixedDataSecHalf.to_csv("mixedDataSecHalf.csv")

# print(mixedDataFirstHalf.sample(n=10))
# print(mixedDataFirstHalf.sample(n=10))
exit()
