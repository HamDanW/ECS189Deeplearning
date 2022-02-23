from code.base_class.dataset import dataset

from string import punctuation

from pathlib import Path


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        if self.dataset_source_file_name == 'text_classification/':
            # set up the file paths

            train_pos_reviews = Path(
                self.dataset_source_folder_path / self.dataset_source_file_name / 'train/' / 'pos').glob('*')
            train_neg_reviews = Path(
                self.dataset_source_folder_path / self.dataset_source_file_name / 'train/' / 'neg').glob('*')

            test_pos_reviews = Path(
                self.dataset_source_folder_path / self.dataset_source_file_name / 'test/' / 'pos').glob('*')
            test_neg_reviews = Path(
                self.dataset_source_folder_path / self.dataset_source_file_name / 'test/' / 'neg').glob('*')


            train_text_pos = ''
            train_text_neg = ''
            test_text_pos = ''
            test_text_neg = ''

            #Store read in data into respective pos/neg vectors
            train_pos_vec = []
            train_neg_vec = []
            test_pos_vec = []
            test_neg_vec = []

            train_y = []
            test_y = []
            
            # read in the reviews into one string
            for file in train_pos_reviews:
                #Get file contents
                train_file = open(file, 'rt', encoding="utf-8")
                train_pos_vec.append(train_file.read())
                train_file.close()

                #Parse file name for label
                filename = file.name.split('_')
                train_y.append(filename[1])
            print("Train Pos Loading Done")
            for file in train_neg_reviews:
                train_file = open(file, 'rt', encoding="utf-8")
                train_neg_vec.append(train_file.read())
                train_file.close()

                #Parse file name for label
                filename = file.name.split('_')
                train_y.append(filename[1])

            print("Train Neg Loading Done")
            for file in test_pos_reviews:
                test_file = open(file, 'rt', encoding="utf-8")
                test_pos_vec.append(test_file.read())
                test_file.close()

                #Parse file name for label
                filename = file.name.split('_')
                test_y.append(filename[1])

            print("Test Pos Loading Done")
            for file in test_neg_reviews:
                test_file = open(file, 'rt', encoding="utf-8")
                test_neg_vec.append(test_file.read())
                test_file.close()

                #Parse file name for label
                filename = file.name.split('_')
                test_y.append(filename[1])

            print("Test Neg Loading Done")
            print('loaded data')
            

            print('cleaning data...')

            # clean the text
            train_all_reviews = []
            all_words = []
            test_all_reviews = []

            for i in range(0,len(train_pos_vec)):
                train_pos_vec[i] = train_pos_vec[i].lower()
                train_pos_vec[i] = "".join([c for c in train_pos_vec[i] if c not in punctuation])
                train_all_reviews.append(train_pos_vec[i].split("\n")[0])
                train_pos_vec[i] = train_pos_vec[i].split(' ')
            
                #Create vocab dictionary
                for word in train_pos_vec[i]:
                    if word not in all_words:
                        all_words.append(word)
            print('train pos list done cleaning')

            for i in range(0,len(train_neg_vec)):
                train_neg_vec[i] = train_neg_vec[i].lower()
                train_neg_vec[i] = "".join([c for c in train_neg_vec[i] if c not in punctuation])
                train_all_reviews.append(train_neg_vec[i].split("\n")[0])
                train_neg_vec[i] = train_neg_vec[i].split(' ')

                #Create vocab dictionary
                for word in train_neg_vec[i]:
                    if word not in all_words:
                        all_words.append(word)

            print('train neg list done cleaning')
            
            for i in range(0,len(test_pos_vec)):
                test_pos_vec[i] = test_pos_vec[i].lower()
                test_pos_vec[i] = "".join([c for c in test_pos_vec[i] if c not in punctuation])
                test_all_reviews.append(test_pos_vec[i].split("\n")[0])
                test_pos_vec[i] = test_pos_vec[i].split(' ')
            
                #Create vocab dictionary
                for word in test_pos_vec[i]:
                    if word not in all_words:
                        all_words.append(word)
            print('test pos list done cleaning')

            for i in range(0,len(test_neg_vec)):
                test_neg_vec[i] = test_neg_vec[i].lower()
                test_neg_vec[i] = "".join([c for c in test_neg_vec[i] if c not in punctuation])
                test_all_reviews.append(test_neg_vec[i].split("\n")[0])
                test_neg_vec[i] = test_neg_vec[i].split(' ')
            
                #Create vocab dictionary
                for word in test_neg_vec[i]:
                    if word not in all_words:
                        all_words.append(word)
            print('test neg list done cleaning')

            print('cleaned data')
            print('Pos List Len: ' + str(len(train_pos_vec)))
            print('Neg List Len: ' + str(len(train_neg_vec)))
            print('All Train reviews List Len: ' + str(len(train_all_reviews)))
            print('All Test reviews List Len: ' + str(len(test_all_reviews)))
            print('All words List Len: ' + str(len(all_words)))

            print('All words List : ' + str(all_words))



            #Encode Train Sentences
            all_encoded_train_sents = []
            all_encoded_test_sents = []

            for review in train_all_reviews:
                #Store encoded form of sentence
                encoded_sent = []
                #Convert review from string format to list format
                words_in_sent = review.split(' ')
                #Assuming train_all_words is vocab dictionary and all words appear once in train_all_words
                #Use index of train_all_words as the numerical mapping for words
                #If word in words_in_sent = train_all_words[i], append i to encoded_sent

                for word in words_in_sent:
                    for i in range(0, len(all_words)):
                        if word is all_words[i]:
                            encoded_sent.append(i)
                            break
                all_encoded_train_sents.append(encoded_sent)
            
            for review in test_all_reviews:
                #Store encoded form of sentence
                encoded_sent = []
                #Convert review from string format to list format
                words_in_sent = review.split(' ')
                print('words in sent: ' + str(words_in_sent))
                break
                #Assuming train_all_words is vocab dictionary and all words appear once in train_all_words
                #Use index of train_all_words as the numerical mapping for words
                #If word in words_in_sent = train_all_words[i], append i to encoded_sent
                for word in words_in_sent:
                    for i in range(0, len(all_words)):
                        if word is all_words[i]:
                            encoded_sent.append(i)
                            break
                all_encoded_test_sents.append(encoded_sent)

            print('Train Y Len: ' + str(len(train_y)))
            print('Test Y Len: ' + str(len(test_y)))

            print('test: ' + str(all_encoded_train_sents[0]))

            #Return Encoded versions of train and test reviews

            #return train_all_reviews, train_all_words, test_all_reviews, test_all_words
            return all_encoded_train_sents, train_y, all_encoded_test_sents, test_y

        
