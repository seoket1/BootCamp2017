from scipy import linalg as la
import numpy as np
from matplotlib import pyplot as plt
import cmath
import math
from scipy.sparse import dok_matrix

# Problem 1
def adjacency(filename):
    # Open `matrix.txt' for read-only
    with open(filename, 'r') as myfile:
        contents = []
        for line in myfile:
            contents.append(line)
    
    # strip() removes trailing whitespace from a line.
    # split() returns a list of the space-separated pieces of the line.
    transition = []
    for i in range(1, len(contents), 1):
        transition.append(contents[i].strip().split())

    int_transition = np.zeros((len(transition),2))
    for i in range(len(transition)):
        for j in range(len(transition[i])):
            int_transition[i,j] = int(transition[i][j])

    # making adjacency matrix
    name_list = np.unique(int_transition)
    adjacency = np.zeros((len(name_list),len(name_list)))

    temp = []
    for i in range(len(name_list)):
        for j in range(len(int_transition[:,0])):
            if name_list[i] == int_transition[j, 0]:
                temp.append(int_transition[j, 0])
        for k in range(len(temp)):
            for l in range(len(int_transition)):
                if temp[k] == int_transition[l, 0]:
                    adjacency[int(temp[k]), int(int_transition[l, 1])] = 1   
        temp = []

    dok_adjaceny = dok_matrix(adjacency)
    return dok_adjaceny, adjacency
    
dok_adjaceny, adjacency = adjacency('C:/Users/suket/Desktop/Homeworks/Computation/Week4/Problem1/matrix.txt')
    
    
# Problem 2
def get_K(adjacency):
    D = np.zeros((len(adjacency), len(adjacency)))
    
    # getting early version of D
    for i in range(len(adjacency)):
        count = 0
        for j in range(len(adjacency)):
            if adjacency[i,j] == 1:
                count += 1
        D[i,i] = count

    # modifying A
    modify_ad = np.copy(adjacency)
    for i in range(len(D)):
        if D[i,i] == 0:
            for j in range(len(D)):
                modify_ad[i,j] = 1
        else:
            print("no sink in row %d"%(i))

    # getting real D
    for i in range(len(adjacency)):
        count = 0
        for j in range(len(adjacency)):
            if modify_ad[i,j] == 1:
                count += 1
        D[i,i] = count

    K = (np.linalg.inv(D) @ modify_ad).T
    print("ok. got the K\n")
    print(K)
    return K

K = get_K(adjacency)


# Problem 3
def find_P(adjacency, N=None, tol=1e-5, d = .85):
    
    if N == None:
        K = get_K(adjacency)
        N = len(K)
    else:
        K = get_K(adjacency[:N, :N])
    p0 = np.random.rand(N)
    dist = 1

    I = np.eye(N)
    print(la.solve(I-d*K, ((1-d)/N)*np.ones(N)))
    while  dist > tol:
        p1 = np.copy(p0)
        p1 = d*K@(p0.T) + ((1-d)/N)*np.ones(N)
        dist = np.linalg.norm(p1-p0)
        p0 = np.copy(p1)
    
    print(p1)
    return p1


p1 = find_P(adjacency, None, tol=1e-5, d = .85)



# Problem 4
def find_P_eigen(adjacency, N=None):
    d = .85
    if N == None:
        K = get_K(adjacency)
        N = len(K)
    else:
        K = get_K(adjacency[:N, :N])

    p0 = np.random.rand(N)
    ein_val, eig_vec =  la.eig(d*K + ((1-d)/N)*np.ones((N,N)))
    p1 = eig_vec[:,0] / np.sum(eig_vec[:,0])
    return p1

print(find_P_eigen(adjacency, N=None))

# Problem 5

with open('C:/Users/suket/Desktop/Homeworks/Computation/Week4/Problem1//ncaa2013.csv', 'r') as ncaafile:
    ncaafile.readline() #reads and ignores the header line
    teams = []
    for line in ncaafile:
        teams.append(line.strip().split(',')) #split on commas
    
    npdata = np.array(teams)

    name_list = np.unique(npdata)
    # making index
    index = np.zeros(len(name_list))
    for i in range(len(name_list)):
        index[i] = int(i)
        
    # making ordered data with index
    ordered_data = np.column_stack((name_list.T, index.T))
    
    # For losing team --> winning team. for my convinience to use previous codes.
    temp_data = np.copy(npdata)
    npdata[:,0] = temp_data[:,1]
    npdata[:,1] = temp_data[:,0]  
    
    # making adjacency matrix
    adjacency = np.zeros((len(name_list),len(name_list)))
    temp = []
    for i in range(len(name_list)):
        for j in range(len(npdata[:,0])):
            if name_list[i] == npdata[j, 0]:
                temp.append(npdata[j, 1])
        for k in range(len(temp)):
            for l in range(len(ordered_data)):
                if temp[k] == ordered_data[l, 0]:
                    adjacency[i, int(float(ordered_data[l, 1]))] = 1   
        temp = []



p1_for_winning_team = find_P(adjacency, N=None, d = .7)


# making index
index_for_finding_team = np.zeros(len(p1_for_winning_team))
for i in range(len(p1_for_winning_team)):
    index_for_finding_team[i] = int(i)

p1_with_index = np.column_stack((p1_for_winning_team.T, index_for_finding_team.T))

order = np.argsort(p1_with_index[:,0])
ordered_p1 = np.take(p1_with_index[:,0], order)

ordered_p1 = ordered_p1[::-1] 

print("\n\n Problem 5\n")

for i in range(5):
    loc = np.where(ordered_p1[i] == p1_for_winning_team)
    print(ordered_data[loc,0])




























    
    