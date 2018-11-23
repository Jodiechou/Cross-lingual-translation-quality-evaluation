import gensim
import sys
from gensim import corpora, models, similarities
import numpy as np
import scipy.optimize
from scipy import spatial
import time
import itertools
import logging
import MeCab
import nltk
import csv
from collections import Counter
from scipy.optimize import linprog

class MinSumWMD:

    def __init__(self):
        self.model = gensim.models.KeyedVectors.load_word2vec_format('ja-en.txt')

    # Convert the tokens to word vectors
    def convert(self, token_ja, token_en):
        vectors_ja = [self.model[w] for w in token_ja]
        vectors_en = [self.model[w] for w in token_en]
        return vectors_ja, vectors_en

    # Add corresponding Euclidean distance to a_ub matrix
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((np.array(x1) - np.array(x2))**2))
    
    def objective_function(self, token_ja, token_en):
        
        vector_ja, vector_en = self.convert(token_ja, token_en)

        count_ja = len(token_ja)
        count_en = len(token_en)
        #t_sum = []
            
        row = count_ja * count_en
        column = count_en + count_ja * count_en
            
        distance = []
        for sub_vector_ja in vector_ja:
            for sub_vector_en in vector_en:
                distance_temp = self.euclidean_distance(sub_vector_ja, sub_vector_en)
                distance.append(distance_temp)
        distance = np.array(distance)

           
        # Generate C_matrix
        c_matrix = np.zeros((column))
        for j in range(count_en):
            c_matrix[j] = 1
            
        # Generate A_ub
        a_ub = np.zeros((row, column))
        for n in range(row):
            a_ub[n][n%count_en] = 1
            a_ub[n][n+count_en] = -1 * distance[n]
           

        # Generate B_ub
        b_ub = np.zeros((row))
            
        t = np.zeros((row))
        res = linprog(c_matrix, a_ub, b_ub, method='interior-point')  
        t = res.x[count_en:]
        n = 0
        t_sum = np.zeros((count_ja))
        for m in range(count_ja):
            t_sum[m] = sum(t[n: n+count_en])
            #t_sum.append(t_sum_temp)
            n += count_en

       
        # Generate B_eq
        times_count = []
        freq_ja = Counter(token_ja)   

        # Calculate outgoing flow from Japanese to English
        weight_d = np.zeros((count_ja))
        times_ja = [freq_ja.get(w_ja) for w_ja in token_ja]
            
        for d in range(count_ja):
            ja = times_ja[d]
            d_ja = ja / count_ja
            weight_d[d] = d_ja
        print("weight_d", weight_d)

        t_matrix = np.zeros((count_ja, count_en))
        s = 0
        for p in range(count_ja):
            for q in range(count_en):
                t_matrix[p][q] = t[s+q] * weight_d[p] / t_sum[p]
            s += count_en

        cost_temp =  distance * t

        sum_min_cost_temp = []
        sum_min_cost = []
        cost = np.zeros((count_ja, count_en))
        r = 0
        for s in range (count_ja):
            cost[s] = cost_temp[r:r+count_en]
            r += count_en
        print("cost",cost)

        min_cost_temp = cost.min(axis=1)
        print("min_cost_temp", min_cost_temp)

        sum_min_cost = min_cost_temp.sum()
        print("sum_min_cost", sum_min_cost)

        """
        sum_min_cost.append(sum_min_cost_temp)
        print("sum_min_cost", sum_min_cost)
        """

        
        print("token_ja", token_ja)
        print("token_en", token_en)
        print("t_sum", t_sum)
        print("t", t)
        print("t_matrix", t_matrix)
        print("distance", distance)
        print("cost_temp", cost_temp)

        


        return sum_min_cost



if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    min_wmd = MinSumWMD()
    mecab = MeCab.Tagger("-Owakati")
    t_score =[]
    with open("results.txt", "w") as res_file:

        with open("filtered.csv", "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                # Japanese tokenisation

                if row[0].startswith("#"):
                   continue
            
                token_ja = mecab.parse(row[0].lower())
                token_ja = token_ja.strip()
                token_ja = token_ja.split(' ')
            
  
                # English tokenisation
                token_en = nltk.word_tokenize(row[1].lower())
                token_en = np.array(token_en)
                method5 = min_wmd.objective_function(token_ja, token_en)
                res_file.write("%s, %s, %f\n" % (row[0], row[0], method5))
    

    # Save the final results
    
    #np.savetxt("output5_1_50", method5)
    
    print("*******************Complete!********************")

    
