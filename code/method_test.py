import gensim
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

    # Convert the tokens to word vectors
    def convert(self, token_ja, token_en):
        model = gensim.models.KeyedVectors.load_word2vec_format('ja-en.txt')
        vectors_ja = [model[w] for w in token_ja]
        vectors_en = [model[w] for w in token_en]
        return vectors_ja, vectors_en

    # Add corresponding Euclidean distance to a_ub matrix
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((np.array(x1) - np.array(x2))**2))
    
    def objective_function(self, token_ja, token_en):
        c_matrix = []
        a_ub = []
        distance = []
        b_ub = []
        b_eq = []
        a_eq = []
        result = []
        objective_fun = []
        sum_min_tc = []
        vector_ja, vector_en = self.convert(token_ja, token_en)

        print("token_ja", token_ja)
        print("token_en", token_en)

        for i in range(len(token_ja)):
            count_ja = len(token_ja[i])
            count_en = len(token_en[i])
            print("count_ja", count_ja)
            print("count_en", count_en)
            # Initialise coefficient matrix, each row contains n*m variables and each column contains m+n*m variables
            row = count_ja * count_en
            column = count_en + count_ja * count_en
            
            distance = []
            for sub_vector_ja in vector_ja[i]:
                for sub_vector_en in vector_en[i]:
                    distance_temp = self.euclidean_distance(sub_vector_ja, sub_vector_en)
                    distance.append(distance_temp)
           
            # Generate C_matrix
            c_matrix = np.zeros((column))
            for j in range(len(token_en[i])):
                c_matrix[j] = 1
            
            # Generate A_ub
            a_ub = np.zeros((row, column))
            for n in range(row):
                a_ub[n][n%count_en] = 1
                a_ub[n][n+count_en] = -1 * distance[n]
           

            # Generate B_ub
            b_ub = np.zeros((1, row))
           
            # Generate B_eq
            times_count = []
            freq_ja = Counter(token_ja[i])
            freq_en = Counter(token_en[i])
            

            """
            # Regard the sums of Tij as 1
            b_eq_temp = np.zeros((1, count_ja+count_en))
            for m in range(count_ja+count_en):
                b_eq_temp[0][m] = 1
            b_eq.append(b_eq_temp)
            """

            # Calculate outgoing flow from Japanese to English
            b_eq = np.zeros((1, count_ja+count_en))
            times_ja = [freq_ja.get(w_ja) for w_ja in token_ja[i]]
            
            for d in range(count_ja):
                ja = times_ja[d]
                d_ja = ja / count_ja
                b_eq[0][d] = d_ja

            # Calculate incoming flow from Japanese to English
            times_en = [freq_en.get(w_en) for w_en in token_en[i]]
            for e in range(count_en):
                en = times_en[e]
                d_en = en / count_en
                b_eq[0][e+count_ja] = d_en
            
            
            # Generate A_eq
            a_eq = np.zeros((count_ja+count_en, column))
            count_en11 = count_en
            for f in range(count_ja):
                for g in range(count_en):
                    a = g + count_en11
                    a_eq[f][a] = 1
                count_en11 += count_en

            for k in range(count_en):
                l = 0
                count_en12 = count_en
                while l < count_ja:
                    b = k + count_en12
                    a_eq[k+count_ja][b] = 1
                    count_en12 += count_en
                    l += 1
           
            
            t = np.zeros((row))
            tc_sum = []
            print(c_matrix.shape)
            print(a_ub.shape)
            res = linprog(c_matrix, a_ub, b_ub, a_eq, b_eq, method='simplex')
            #print("result", res)
            #result.append(res.fun)
            t = res.x[count_en:]
            print("result", res)
            print("t", t)
            dist = np.array(distance)
            print("dist", dist)

            tc_temp = dist * t
            print("tc_temp", tc_temp)

            sum_min_tc_temp = []
            tc = np.zeros((count_ja, count_en))
            p = 0
            for q in range (count_ja):
                tc[q] = tc_temp[p:p+count_en]
                p += count_en
            print("tc",tc)

            min_tc_temp = tc.min(axis=1)
            print("min_tc_temp", min_tc_temp)

            sum_min_tc_temp = min_tc_temp.sum()
            print("sum_min_tc_temp", sum_min_tc_temp)

            sum_min_tc.append(sum_min_tc_temp)
            print("sum_min_tc", sum_min_tc)

        return sum_min_tc



if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    temp_ja = []
    token_ja = []
    vectors_ja = []
    temp_en = []
    token_en = []
    with open("method_test.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # Japanese tokenisation
            mecab = MeCab.Tagger("-Owakati")
            temp_ja = mecab.parse(row[0].lower())
            temp_ja = temp_ja.strip()
            temp_ja = temp_ja.split(' ')
            token_ja.append(temp_ja)
  
            # English tokenisation
            temp_en = nltk.word_tokenize(row[1].lower())
            temp_en = np.array(temp_en)
            token_en.append(temp_en)
    
    min_wmd = MinSumWMD()

    # Save the final results
    method5 = min_wmd.objective_function(token_ja, token_en)
    np.savetxt("output5-manual-whole-actualCost-test", method5)
    
    print("*******************Complete!********************")

    
