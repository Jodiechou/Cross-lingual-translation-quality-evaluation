import gensim
from gensim import corpora, models, similarities
import numpy as np
import scipy.optimize
from scipy import spatial
import logging
import MeCab
import nltk
import csv
from collections import Counter
from scipy.optimize import linprog


class SMWMD:

    # Convert the tokens to word vectors
    def convert(self, token_ja, token_en):
        model = gensim.models.KeyedVectors.load_word2vec_format('ja-en.txt')
        vectors_ja = [model[w] for w in token_ja]
        vectors_en = [model[w] for w in token_en]
        return vectors_ja, vectors_en

    # Add corresponding Euclidean distance to a_ub matrix
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((np.array(x1) - np.array(x2))**2))
    
    def coefficient_matrixs(self, token_ja, token_en):
        lp_c_matrix = []
        a_ub = []
        distance = []
        b_ub = []
        distance_matrix = []
        b_eq = []
        a_eq = []
        distance_matrix = []
        vector_ja, vector_en = self.convert(token_ja, token_en)
        
        for i in range(len(token_ja)):
            count_ja = len(token_ja[i])
            count_en = len(token_en[i])
            
            # Initialise coefficient matrix, each row contains n*m variables and each column contains m+n*m variables
            row = count_ja * count_en
            column = count_en + count_ja * count_en
            
            distance = []
            for sub_vector_ja in vector_ja[i]:
                for sub_vector_en in vector_en[i]:
                    distance_temp = self.euclidean_distance(sub_vector_ja, sub_vector_en)
                    distance.append(distance_temp)
            distance_matrix.append(np.array(distance))

            # Generate C_matrix
            j = 0 
            q = 0
            lp_c_matrix_temp = []
            while j < count_en:
                lp_c_matrix_temp.append(1)
                j += 1
            while q < count_ja*count_en:
                lp_c_matrix_temp.append(0)
                q += 1
            lp_c_matrix.append(np.array(lp_c_matrix_temp))

        
            # Generate A_ub
            
            a_ub_temp = np.zeros((row, column))
            for n in range(row):
                a_ub_temp[n][n%count_en] = 1
                a_ub_temp[n][n+count_en] = -1 * distance[n]
            a_ub.append(a_ub_temp)


            # Generate B_ub
            b1 = 0
            b_ub_temp = []
            while b1 < count_en*count_ja:
                b_ub_temp.append(0)
                b1 +=1
            b_ub.append(np.array(b_ub_temp))
            

            # Generate B_eq
            times_count = []
            c_ja = Counter(token_ja[i])
            c_en = Counter(token_en[i])
            

            # Calculate outgoing flow from Japanese to English
            b_eq_temp = []

            # get the times of a word appearing in a document; w_ja is the word, c_ja is the counter dictionary
            # For example:
            # c_ja Counter({'私': 1, 'は': 1, '学校': 1, 'が': 1, '好き': 1, 'です': 1})
            # [c_ja.get(w_ja) for w_ja in token_ja[i]], do: for word in token Japanese words get the frequency of the word
            times_ja = [c_ja.get(w_ja) for w_ja in token_ja[i]]
            
            for d in range(count_ja):
                d_ja = times_ja[d] / count_ja
                b_eq_temp.append(d_ja)
                

            # Calculate incoming flow from Japanese to English
            times_en = [c_en.get(w_en) for w_en in token_en[i]]
            for e in range(count_en):
                d_en = times_en[e] / count_en
                b_eq_temp.append(d_en)
                #print("2", b_eq_temp)
            b_eq.append(np.array(b_eq_temp))
            
            # Generate A_eq
            a_eq_temp = np.zeros((count_ja+count_en, column))
            count_en11 = count_en
            for f in range(count_ja):
                for g in range(count_en):
                    a = g + count_en11
                    a_eq_temp[f][a] = 1
                count_en11 += count_en

            for k in range(count_en):
                l = 0
                count_en12 = count_en
                while l < count_ja:
                    b = k + count_en12
                    a_eq_temp[k+count_ja][b] = 1
                    count_en12 += count_en
                    l += 1
            a_eq.append(a_eq_temp)
        
        return lp_c_matrix, a_ub, b_ub, a_eq, b_eq, distance_matrix
        

    def ojective_function(self, token_ja, token_en):
        
        lp_c_matrix, a_u, b_u, a_e, b_e, dist = self.coefficient_matrixs(token_ja, token_en)
        z_result = []
        t = []
        tc_sum = []

        for h in range(len(token_ja)):
            count_ja = len(token_ja[h])
            count_en = len(token_en[h])
            c = lp_c_matrix[h]
            A_ub = a_u[h]
            b_ub = b_u[h]
            A_eq = a_e[h]
            b_eq = b_e[h]
            
            res = linprog(c, A_ub, b_ub, A_eq, b_eq, method='interior-point')
            z_result.append(res.fun)
            t.append(res.x[count_en:])
            
        
        dist = np.array(dist)   
        t = np.array(t)
        tc_temp = dist * t

        sum_min_tc = []
        for p in range(len(token_ja)):
            count_ja = len(token_ja[p])
            count_en = len(token_en[p])
            tc = np.zeros((count_ja, count_en))
            q = 0
            for p1 in range(count_ja):
                tc[p1] = tc_temp[p][q:q+count_en]
                q += count_en
            tc_transposed = tc.transpose()
            min_tc_temp = tc_transposed.min(axis=1)
            sum_min_temp = min_tc_temp.sum()
            sum_min_tc.append(sum_min_temp)


            np.set_printoptions(threshold=None)
            
            
            print("tc_transposed", tc_transposed)
            print("minimum value", min_tc_temp)
            print("sum of the minimum values",sum_min_temp)
            
        print("list of the sum of minimun values", sum_min_tc)

        return z_result, sum_min_tc
       




if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    temp_ja = []
    token_ja = []
    vectors_ja = []
    temp_en = []
    token_en = []

    # read the original corpus
    with open("corpus.csv", "r", encoding="utf-8-sig") as f:
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

    
    sm_wmd = SMWMD()
    
    # Save the final results
    method5 = sm_wmd.ojective_function(token_ja, token_en)
    np.savetxt("output5-manualtranslation", method5[0])
    np.savetxt("output5-manualtranslation", method5[1])
   
    print("*******************Complete!********************")
