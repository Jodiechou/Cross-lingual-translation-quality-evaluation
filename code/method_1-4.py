import MeCab
import nltk
import csv
import gensim
import numpy as np
import logging
 
class QualityEvaluation(object):
    # Convert the tokens to word vectors
    def convert(self, token_ja, token_en):
        model = gensim.models.KeyedVectors.load_word2vec_format('ja-en.txt')
        vectors_ja = [model[w] for w in token_ja]
        vectors_en = [model[w] for w in token_en]
        return vectors_ja, vectors_en
   
    # Method 1 
    def method1_averaged(self, token_ja, token_en):
        sentence_ja_average, sentence_en_average = self.convert(token_ja, token_en)
        averaged_ja = []
        averaged_en = []
        similarity_averaged = []
        for element_ja_average in sentence_ja_average:
            averaged_ja.append(np.mean(element_ja_average, axis=0))
        for element_en_average in sentence_en_average:
            averaged_en.append(np.mean(element_en_average, axis=0))
        for n in range(len(averaged_en)):
            similarity_temp_average = np.dot(averaged_ja[n], averaged_en[n])/(np.linalg.norm(averaged_ja[n])*(np.linalg.norm(averaged_en[n])))
            similarity_averaged.append(similarity_temp_average)

        #print("----------------------Method1 Averaged------------------------")
        #print("Method1 similarity averaged: ", similarity_averaged)
        return similarity_averaged

   # Method 2 
    def method2_max_source_centered(self, token_ja, token_en):
        temp_source = 0
        max_temp_source = 0
        k_source = 0
        max_similarity_source = []
        similarity_score_source = []
        source_centered = []
        sentence_ja_source, sentence_en_source = self.convert(token_ja, token_en)
        sentence_ja_source = np.array(sentence_ja_source)
        sentence_en_source = np.array(sentence_en_source)

        for i in range(len(sentence_ja_source)):
            for n in range(len(sentence_ja_source[i])):
                for m in range(len(sentence_en_source[i])):
                    similarity_source_temp = np.dot(sentence_ja_source[i][n], sentence_en_source[i][m])/(np.linalg.norm(sentence_ja_source[i][n])*(np.linalg.norm(sentence_en_source[i][m])))
                    if max_temp_source < similarity_source_temp:
                        max_temp_source = similarity_source_temp
                
                max_similarity_source.append(max_temp_source)
                max_temp_source = 0
               
            temp_source = len(sentence_ja_source[i])
            k_source = temp_source
            if similarity_score_source == []:
                similarity_score_source.append(max_similarity_source[0:k_source])
            else:
                similarity_score_source.append(max_similarity_source[k_source+1:k_source+temp_source+1])

        for element_score_source in similarity_score_source:
            source_centered.append(np.mean(element_score_source, axis=0))

        #print("------------Method2 Max similarity source centered------------")
        #print("Method2 Max similarity source centered: ", source_centered)
        return source_centered


    # Method 3
    def method3_max_target_centered(self, token_ja, token_en):
        temp_target = 0
        max_temp_target = 0
        #k_target = 0
        max_similarity_target = []
        similarity_score_target = []
        target_centered = []
        sentence_ja_target, sentence_en_target = self.convert(token_ja, token_en)
        sentence_ja_target = np.array(sentence_ja_target)
        sentence_en_target = np.array(sentence_en_target)

        for i in range(len(sentence_en_target)):
            for n in range(len(sentence_en_target[i])):
                for m in range(len(sentence_ja_target[i])):
                    similarity_target_temp = np.dot(sentence_en_target[i][n], sentence_ja_target[i][m])/(np.linalg.norm(sentence_en_target[i][n])*(np.linalg.norm(sentence_ja_target[i][m])))
                    if max_temp_target < similarity_target_temp:
                        max_temp_target = similarity_target_temp
                
                max_similarity_target.append(max_temp_target)
                max_temp_target = 0
               
            temp_target = len(sentence_en_target[i])
            k_target = temp_target
            if similarity_score_target == []:
                similarity_score_target.append(max_similarity_target[0:k_target])
            else:
                similarity_score_target.append(max_similarity_target[k_target+1:k_target+temp_target+1])

        for element_score_target in similarity_score_target:
            target_centered.append(np.mean(element_score_target, axis=0))

        #print("------------Method3 Max similarity target centered------------")
        #print("Method3 Max similarity target centered: ", target_centered)
        return target_centered

    # Method 4 
    def method4_wmd(self, token_ja, token_en):
        model = gensim.models.KeyedVectors.load_word2vec_format('ja-en.txt')
        distance = []
        similarity = []
        for i in range(len(token_ja)):
            distance_temp = model.wmdistance(token_ja[i], token_en[i])
            distance.append(distance_temp)
            print("distance_temp", distance_temp)
            similarity_temp = 1/distance_temp
            similarity.append(similarity_temp)
        
        return distance, similarity

           

if __name__ == "__main__":
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    temp_ja = []
    token_ja = []
    vectors_ja = []
    temp_en = []
    token_en = []

    with open("google_translation.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # Japanese tokenisation
            mecab = MeCab.Tagger("-Owakati")
            temp_ja = mecab.parse(row[0].lower())
            temp_ja = temp_ja.split(' ')
            token_ja.append(temp_ja)
  
            # English tokenisation
            temp_en = nltk.word_tokenize(row[1].lower())
            temp_en = np.array(temp_en)
            token_en.append(temp_en)
    
    new_token_ja = []
    
    for temp_ja in token_ja:
        temp_ja.remove('\n')
        temp_ja = np.array(temp_ja)
        new_token_ja.append(temp_ja)

    token_ja = np.array(new_token_ja)
    token_en = np.array(token_en)
    

    quality_evaluation = QualityEvaluation()
    
    # Save the final results
    np.savetxt("output1-manual-whole", quality_evaluation.method1_averaged(token_ja, token_en))
    np.savetxt("output2-manual-whole", quality_evaluation.method2_max_source_centered(token_ja, token_en))
    np.savetxt("output3-manual-whole", quality_evaluation.method3_max_target_centered(token_ja, token_en))

    method4 = quality_evaluation.method4_wmd(token_ja, token_en)
    np.savetxt("output4-manual-whole-distance", method4[0])
    np.savetxt("output4-manual-whole-similarity", method4[1])
    print("Complete!")




