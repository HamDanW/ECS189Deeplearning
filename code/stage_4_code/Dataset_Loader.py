import string
from code.base_class.dataset import dataset

from string import punctuation

from pathlib import Path

import numpy as np

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    load_from_files = True
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        if self.dataset_source_file_name == 'text_classification/':
            train_all_reviews = []
            all_words = []
            test_all_reviews = []
            train_y = []
            test_y = []
            all_encoded_train_sents = []
            all_encoded_test_sents = []


            if self.load_from_files:
                #Load train_all.txt into train_all_reviews[]
                train_file_loc = Path('script/stage_4_script/train_all.txt')
                train_file = open(train_file_loc, 'rt', encoding='utf-8')
                #1 review per line
                for line in train_file:
                    review = line.split('\n')[0]
                    review = review.split(' ')
                    train_all_reviews.append(review)
                print('train_all.txt finished loading')

                #Load test_all.txt into test_all_reviews[]
                test_file_loc = Path('script/stage_4_script/test_all.txt')
                test_file = open(test_file_loc, 'rt', encoding='utf-8')
                #1 review per line
                for line in test_file:
                    review = line.split('\n')[0]
                    review = review.split(' ')
                    test_all_reviews.append(review)
                print('test_all.txt finished loading')

                #Load trainY_labels.txt into trainY_labels
                trainY_file_loc = Path('script/stage_4_script/trainY_labels.txt')
                trainY_file = open(trainY_file_loc, 'rt', encoding='utf-8')
                for line in trainY_file:
                    review = line.split('\n')[0]
                    train_y.append(int(review))
                print('trainY_labels.txt finished loading')

                #Load testY_labels.txt into testY_labels
                testY_file_loc = Path('script/stage_4_script/testY_labels.txt')
                testY_file = open(testY_file_loc, 'rt', encoding='utf-8')
                for line in testY_file:
                    review = line.split('\n')[0]
                    test_y.append(int(review))
                print('testY_labels.txt finished loading')

                #Load vocab.txt into all_words[]
                vocab_file_loc = Path('script/stage_4_script/vocab.txt')
                vocab_file = open(vocab_file_loc, 'rt', encoding='utf-8')
                for line in vocab_file:
                    word = line.split('\n')[0]
                    all_words.append(word)
                print('vocab.txt finished loading')

                
                #Load train encoded
                train_encoded_file_loc = Path('script/stage_4_script/train_encoded.txt')
                train_encoded_file = open(train_encoded_file_loc, 'rt', encoding='utf-8')
                for line in train_encoded_file:
                    encoded = line.split('\n')[0].split(' ')
                    
                    temp = []
                    for num in encoded:
                        temp.append(int(num))
                    all_encoded_train_sents.append(temp)
                print('train_encoded.txt finished loading')

                #Load test encoded
                test_encoded_file_loc = Path('script/stage_4_script/test_encoded.txt')
                test_encoded_file = open(test_encoded_file_loc, 'rt', encoding='utf-8')
                for line in test_encoded_file:
                    encoded = line.split('\n')[0].split(' ')
                    temp = []
                    for num in encoded:
                        temp.append(int(num))
                    all_encoded_test_sents.append(temp)
                print('test_encoded.txt finished loading')
                
                
                         
                

                


            else:
                # set up the file paths

                train_pos_reviews = Path(
                    self.dataset_source_folder_path / self.dataset_source_file_name / 'train/' / 'pos').glob('*')
                train_neg_reviews = Path(
                    self.dataset_source_folder_path / self.dataset_source_file_name / 'train/' / 'neg').glob('*')

                test_pos_reviews = Path(
                    self.dataset_source_folder_path / self.dataset_source_file_name / 'test/' / 'pos').glob('*')
                test_neg_reviews = Path(
                    self.dataset_source_folder_path / self.dataset_source_file_name / 'test/' / 'neg').glob('*')

                #If data is stored in the following text files, load from text files instead
                #Files are 'test_all.txt', 'train_all.txt', trainY_labels.txt', 'testY_labels.txt', 'vocab.txt'



                train_text_pos = ''
                train_text_neg = ''
                test_text_pos = ''
                test_text_neg = ''

                #Store read in data into respective pos/neg vectors
                train_pos_vec = []
                train_neg_vec = []
                test_pos_vec = []
                test_neg_vec = []

                
                
                # read in the reviews into one string
                for file in train_pos_reviews:
                    #Get file contents
                    train_file = open(file, 'rt', encoding="utf-8")
                    train_pos_vec.append(train_file.read())
                    
                    #Parse file name for label
                    # Let <= 4 be negative (0) and >= 7 be positive (1)
                    filename = str(train_file.name.split('_')[-1]).split('.')
                    if int(filename[0]) <= 4:
                        train_y.append('0')
                    elif int(filename[0]) >= 7:
                        train_y.append('1')
                    #train_y.append(filename[0])
                    train_file.close()
                print("Train Pos Loading Done")


                for file in train_neg_reviews:
                    train_file = open(file, 'rt', encoding="utf-8")
                    train_neg_vec.append(train_file.read())
                    

                    #Parse file name for label
                    filename = str(train_file.name.split('_')[-1]).split('.')
                    if int(filename[0]) <= 4:
                        train_y.append('0')
                    elif int(filename[0]) >= 7:
                        train_y.append('1')
                    #train_y.append(filename[0])
                    train_file.close()
                print("Train Neg Loading Done")

                for file in test_pos_reviews:
                    test_file = open(file, 'rt', encoding="utf-8")
                    test_pos_vec.append(test_file.read())

                    #Parse file name for label
                    filename = str(test_file.name.split('_')[-1]).split('.')
                    if int(filename[0]) <= 4:
                        test_y.append('0')
                    elif int(filename[0]) >= 7:
                        test_y.append('1')

                    #test_y.append(filename[0])
                    test_file.close()

                print("Test Pos Loading Done")
                for file in test_neg_reviews:
                    test_file = open(file, 'rt', encoding="utf-8")
                    test_neg_vec.append(test_file.read())

                    #Parse file name for label
                    filename = str(file.name.split('_')[-1]).split('.')
                    if int(filename[0]) <= 4:
                        test_y.append('0')
                    elif int(filename[0]) >= 7:
                        test_y.append('1')
                    #test_y.append(filename[0])
                    test_file.close()

                print("Test Neg Loading Done")
                print('loaded data')

                # Save labels in text file
                train_y_file = open('script/stage_4_script/trainY_labels.txt', 'r+', encoding='utf-8')
                for label in train_y:
                    train_y_file.write(label + '\n')
                train_y_file.close()

                test_y_file = open('script/stage_4_script/testY_labels.txt', 'r+', encoding='utf-8')
                for label in test_y:
                    test_y_file.write(label + '\n')    
                test_y_file.close()
                

                print('cleaning data...')

                # clean the text
                stop_words = set(stopwords.words('english'))
                
                ps = PorterStemmer()

                print('Cleaning Train Pos')
                for i in range(0,len(train_pos_vec)):
                    #For each element, first change to lowercase
                    train_pos_vec[i] = train_pos_vec[i].lower()

                    #Next tokenize and remove puncuation and newlines
                    train_pos_vec[i] = word_tokenize(train_pos_vec[i])
                    train_pos_vec[i] = [word for word in train_pos_vec[i] if word.isalpha()]

                    #Remove stop words
                    stop_words = stopwords.words('english')
                    train_pos_vec[i] = [word for word in train_pos_vec[i] if not word in stop_words]
                    #review = ' '.join(train_pos_vec[i])

                    #Stem words
                    train_pos_vec[i] = [ps.stem(word) for word in train_pos_vec[i]]

                    #Truncate down to 80 words or pad to 80
                    words_in_sent = [] + train_pos_vec[i]

                    if len(words_in_sent) >= 20:
                        words_in_sent = words_in_sent[0:20]
                    elif len(words_in_sent) < 20:
                        while len(words_in_sent) < 20:
                            words_in_sent.append('-1')


                    review = ' '.join(words_in_sent)
                    train_all_reviews.append(review)

                    '''
                    train_pos_vec[i] = train_pos_vec[i].lower()
                    train_pos_vec[i] = "".join([c for c in train_pos_vec[i] if c not in punctuation])
                    review = train_pos_vec[i].split("\n")[0]
                    review = ' '.join(review)
                    train_all_reviews.append(review)
                    train_pos_vec[i] = train_pos_vec[i].split(' ')
                    '''

                    #Create vocab dictionary
                    for word in words_in_sent:
                        if word not in all_words:
                                all_words.append(word)


                print('train pos list done cleaning')

                #Saving Cleaned Train Pos Data
                print('Saving Train Pos')
                word_string = ''
                for review in train_pos_vec:
                    joined = ' '.join(review)
                    word_string = word_string + joined + '\n'
                train_pos_file = open('script/stage_4_script/train_pos.txt', 'r+', encoding="utf-8")
                train_pos_file.write(word_string)
                train_pos_file.close()
                print('train pos finished saving')

                

                print('Cleaning Train Neg')
                for i in range(0,len(train_neg_vec)):
                    #For each element, first change to lowercase
                    train_neg_vec[i] = train_neg_vec[i].lower()

                    #Next tokenize and remove puncuation and newlines
                    train_neg_vec[i] = word_tokenize(train_neg_vec[i])
                    train_neg_vec[i] = [word for word in train_neg_vec[i] if word.isalpha()]

                    #Remove stop words
                    stop_words = stopwords.words('english')
                    train_neg_vec[i] = [word for word in train_neg_vec[i] if not word in stop_words]
                    #review = ' '.join(train_neg_vec[i])

                    #Stem words
                    train_neg_vec[i] = [ps.stem(word) for word in train_neg_vec[i]]

                    #Truncate down to 80 words or pad to 80
                    words_in_sent = [] + train_neg_vec[i]
                    if len(words_in_sent) >= 20:
                        words_in_sent = words_in_sent[0:20]
                    elif len(words_in_sent) < 20:
                        while len(words_in_sent) < 20:
                            words_in_sent.append('-1')

                    review = ' '.join(words_in_sent)
                    train_all_reviews.append(review)

                    '''
                    train_neg_vec[i] = train_neg_vec[i].lower()
                    train_neg_vec[i] = "".join([c for c in train_neg_vec[i] if c not in punctuation])
                    review = train_neg_vec[i].split("\n")[0]
                    review = ' '.join(review)
                    train_all_reviews.append(review)
                    train_neg_vec[i] = train_neg_vec[i].split(' ')
                    '''

                    #Create vocab dictionary
                    for word in words_in_sent:
                        if word not in all_words:
                                all_words.append(word)


                print('train neg list done cleaning')

                #Saving Cleaned Train Neg Data
                word_string = ''
                for review in train_neg_vec:
                    joined = ' '.join(review)
                    word_string = word_string + joined + '\n'
                train_neg_file = open('script/stage_4_script/train_neg.txt', 'r+', encoding="utf-8")
                train_neg_file.write(word_string)
                train_neg_file.close()
                print('train neg finished saving')
                
                print('Cleaning Test Pos')
                for i in range(0,len(test_pos_vec)):
                    #For each element, first change to lowercase
                    test_pos_vec[i] = test_pos_vec[i].lower()

                    #Next tokenize and remove puncuation and newlines
                    test_pos_vec[i] = word_tokenize(test_pos_vec[i])
                    test_pos_vec[i] = [word for word in test_pos_vec[i] if word.isalpha()]

                    #Remove stop words
                    stop_words = stopwords.words('english')
                    test_pos_vec[i] = [word for word in test_pos_vec[i] if not word in stop_words]
                    #review = ' '.join(test_pos_vec[i])

                    #Stem words
                    test_pos_vec[i] = [ps.stem(word) for word in test_pos_vec[i]]

                    #Truncate down to 80 words or pad to 80
                    words_in_sent = [] + test_pos_vec[i]
                    if len(words_in_sent) >= 20:
                        words_in_sent = words_in_sent[0:20]
                    elif len(words_in_sent) < 20:
                        while len(words_in_sent) < 20:
                            words_in_sent.append('-1')

                    review = ' '.join(words_in_sent)
                    test_all_reviews.append(review)                  
                    '''
                    test_pos_vec[i] = test_pos_vec[i].lower()
                    test_pos_vec[i] = "".join([c for c in test_pos_vec[i] if c not in punctuation])
                    review = test_pos_vec[i].split("\n")[0]
                    review = ' '.join(review)
                    test_all_reviews.append(review)
                    test_pos_vec[i] = test_pos_vec[i].split(' ')
                    '''

                    #Create vocab dictionary
                    for word in words_in_sent:
                        if word not in all_words:
                                all_words.append(word)

                print('test pos list done cleaning')

                #Saving Cleaned Test Pos Data
                word_string = ''
                for review in test_pos_vec:
                    joined = ' '.join(review)
                    word_string = word_string + joined + '\n'
                test_pos_file = open('script/stage_4_script/test_pos.txt', 'r+', encoding="utf-8")
                test_pos_file.write(word_string)
                test_pos_file.close()
                print('test pos finished saving')

                print('Cleaning Test Neg')
                for i in range(0,len(test_neg_vec)):
                    #For each element, first change to lowercase
                    test_neg_vec[i] = test_neg_vec[i].lower()

                    #Next tokenize and remove puncuation and newlines
                    test_neg_vec[i] = word_tokenize(test_neg_vec[i])
                    test_neg_vec[i] = [word for word in test_neg_vec[i] if word.isalpha()]

                    #Remove stop words
                    stop_words = stopwords.words('english')
                    test_neg_vec[i] = [word for word in test_neg_vec[i] if not word in stop_words]
                    #review = ' '.join(test_neg_vec[i])

                    #Stem words
                    test_neg_vec[i] = [ps.stem(word) for word in train_neg_vec[i]]

                    #Truncate down to 100 words or pad to 100
                    words_in_sent = [] + test_neg_vec[i]
                    if len(words_in_sent) >= 20:
                        words_in_sent = words_in_sent[0:20]
                    elif len(words_in_sent) < 20:
                        while len(words_in_sent) < 20:
                            words_in_sent.append('-1')


                    review = ' '.join(words_in_sent)
                    test_all_reviews.append(review)                     
                    '''
                    test_neg_vec[i] = test_neg_vec[i].lower()
                    test_neg_vec[i] = "".join([c for c in test_neg_vec[i] if c not in punctuation])
                    review = test_neg_vec[i].split("\n")[0]
                    review = ' '.join(review)
                    test_all_reviews.append(review)
                    test_neg_vec[i] = test_neg_vec[i].split(' ')
                    '''

                    #Create vocab dictionary
                    for word in words_in_sent:
                        if word not in all_words:
                                all_words.append(word)

                print('test neg list done cleaning')

                #Saving Cleaned Test Neg Data
                word_string = ''
                for review in test_neg_vec:
                    joined = ' '.join(review)
                    word_string = word_string + joined + '\n'
                test_neg_file = open('script/stage_4_script/test_neg.txt', 'r+', encoding="utf-8")
                test_neg_file.write(word_string)
                test_neg_file.close()
                print('test neg finished saving')

                print('Pos List Len: ' + str(len(train_pos_vec)))
                print('Neg List Len: ' + str(len(train_neg_vec)))
                print('All Train reviews List Len: ' + str(len(train_all_reviews)))
                print('All Test reviews List Len: ' + str(len(test_all_reviews)))
                print('All words List Len: ' + str(len(all_words)))

                #print('All words List : ' + str(all_words))

                #Save data in files to increase faster loading

                #Saving Cleaned All Train Data
                word_string = ''
                for review in train_all_reviews:
                    word_string = word_string + str(review) + '\n'
                all_train_file = open('script/stage_4_script/train_all.txt', 'r+', encoding="utf-8")
                all_train_file.write(word_string)
                all_train_file.close()
                print('train all reviews finished saving')

                #Saving Cleaned All Test Data
                word_string = ''
                for review in test_all_reviews:
                    word_string = word_string + str(review) + '\n'
                all_test_file = open('script/stage_4_script/test_all.txt', 'r+', encoding="utf-8")
                all_test_file.write(word_string)
                all_test_file.close()
                print('test all reviews finished saving')

                #Saving vocab
                word_string = ''
                for word in all_words:
                    word_string = word_string + str(word) + '\n'
                vocab_file = open('script/stage_4_script/vocab.txt', 'r+', encoding="utf-8")
                vocab_file.write(word_string)
                vocab_file.close()
                print('vocab finished saving')
                print(len(all_words))

            #return 1,1,1,1

            
                print('Begin Encoding Train')
                #Encode Train Sentences
                for i in range(0, len(train_all_reviews)):
                    words_in_sent = train_all_reviews[i].split(' ')
                    '''
                    if len(words_in_sent) >= 200:
                        words_in_sent = words_in_sent[0:200]
                    elif len(words_in_sent) < 200:
                        while len(words_in_sent) < 200:
                            words_in_sent.append('-1')
                    '''

                    #Assuming train_all_words is vocab dictionary and all words appear once in train_all_words
                    #Use index of train_all_words as the numerical mapping for words
                    #If word in words_in_sent = train_all_words[i], append i to encoded_sent
                    encoded_sent = []
                    for word in words_in_sent:
                        #Store encoded form of sentence
                        if word in all_words:
                            encoded_sent.append(all_words.index(word))
                        elif word == -1:
                            encoded_sent.append(-1)
                    all_encoded_train_sents.append(encoded_sent)
                print('Finished encoding Train')

                #print(all_encoded_test_sents)

                print('Begin Encoding Test')
                #Encode Test Sentences
                for i in range(0, len(test_all_reviews)):
                    words_in_sent = test_all_reviews[i].split(' ')
                    '''
                    if len(words_in_sent) >= 200:
                        words_in_sent = words_in_sent[0:200]
                    elif len(words_in_sent) < 200:
                        while len(words_in_sent) < 200:
                            words_in_sent.append('-1')
                    '''
                    #Assuming train_all_words is vocab dictionary and all words appear once in train_all_words
                    #Use index of train_all_words as the numerical mapping for words
                    #If word in words_in_sent = train_all_words[i], append i to encoded_sent
                    encoded_sent = []
                    for word in words_in_sent:
                        #Store encoded form of sentence
                        if word in all_words:
                            encoded_sent.append(all_words.index(word))
                        elif word == -1:
                            encoded_sent.append(-1)
                    all_encoded_test_sents.append(encoded_sent)
                print('Finished Encoding Test')

                print('Begin saving train encoding')
                #Save train encoded into train_encoded.txt
                word_string = ''
                for review in all_encoded_train_sents:
                    joined = ' '.join(review)
                    word_string = word_string + joined + '\n'
                train_encoded_file = open('script/stage_4_script/train_encoded.txt', 'r+', encoding="utf-8")
                train_encoded_file.write(word_string)
                train_encoded_file.close()
                print('train encoded finished saving')

                print('Begin saving test encoding')
                #Save test encoded into test_encoded.txt
                word_string = ''
                for review in all_encoded_test_sents:
                    joined = ' '.join(review)
                    word_string = word_string + joined + '\n'
                test_encoded_file = open('script/stage_4_script/test_encoded.txt', 'r+', encoding="utf-8")
                test_encoded_file.write(word_string)
                test_encoded_file.close()

            #Return Encoded versions of train and test reviews

            #return train_all_reviews, train_all_words, test_all_reviews, test_all_words
            return all_encoded_train_sents, train_y, all_encoded_test_sents, test_y, all_words

        elif self.dataset_source_file_name == 'text_generation/':
            # set file path
            jokes = Path(self.dataset_source_folder_path / self.dataset_source_file_name / 'data')
            jokes_vec = []

            jokes_file = open(jokes, 'r+', encoding='utf-8')
            # add jokes to vector, get rid of id in beginning
            for line in jokes_file:
                jokes_vec.append(line[line.find(',')+1:].lower())

            # remove first line because it doesn't contain a joke
            jokes_vec.pop(0)

            jokes_vec_clean = []
            for line in jokes_vec:
                curr = ''
                curr_tokens = word_tokenize(line)
                curr += ' '.join([word for word in curr_tokens if word.isalnum()])
                # jokes_vec_clean.append([word for word in curr_tokens if word.isalnum()])
                jokes_vec_clean.append(curr)

            # jokes_vec_clean = sum(jokes_vec_clean, [])

            # create vocab list
            vocab = sorted(set(jokes_vec))
            vocab_clean = []
            for line in vocab:
                curr_tokens = word_tokenize(line)
                vocab_clean.append([word for word in curr_tokens if word.isalnum()])

            vocab_clean = sum(vocab_clean, [])

            print('all jokes list length:', len(jokes_vec_clean))
            print('vocab list length:', len(vocab_clean))

            # encoding
            sequences = []

            for joke in jokes_vec_clean:
                curr_joke = ''.join(joke)
                if len(curr_joke.split(' ')) > 5:
                    for i in range(5, len(curr_joke.split(' '))):
                        seq = curr_joke.split(' ')[i-5:i+1]
                        sequences.append(' '.join(seq))

            print('sequence list size:', len(sequences))

            x = []
            y = []

            for s in sequences:
                x.append(" ".join(s.split()[:-1]))
                y.append(" ".join(s.split()[1:]))

            # create integer-to-token mapping
            int2token = {}
            cnt = 0

            for w in set(" ".join(jokes_vec_clean).split()):
                int2token[cnt] = w
                cnt += 1

            # create token-to-integer mapping
            token2int = {t: i for i, t in int2token.items()}

            def get_integer_seq(seq):
                return [token2int[w] for w in seq.split()]

            # convert text sequences to integer sequences
            x_int = [get_integer_seq(i) for i in x]
            y_int = [get_integer_seq(i) for i in y]

            return x_int, y_int

