import re
import nltk
import unicodedata
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


FILE_PATH = '/home/vudat1710/Downloads/NLP/CQA/file.txt'

class PreprocessData:
	def url_elimination(self, text):
		urls = re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', text)
		output = ''
		for url in urls:
			x = text.find(url)
			if x > 0:
				output += text[:x]
				output += "url "
				text = text[x+len(url) +1:]
		output += text
		return output

	def tokenize(self, text):
		text = self.url_elimination(text)
		return [w.lower() for w in nltk.word_tokenize(text)]
		
	def remove_non_ascii(self, words):
		"""Remove non-ASCII characters from list of tokenized words"""
		new_words = []
		for word in words:
			new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
			new_words.append(new_word)
		return new_words

	def remove_punctuation2(self, words):
		new_words = []
		for word in words:
			temp = word.strip(string.punctuation)
			if temp is not '':
				new_words.append(temp)
		return new_words

	def replace_numbers(self, words):
		"""Replace all interger occurrences in list of tokenized words with textual representation"""
		return [re.sub(r'\d+', '<num>', word) for word in words]

	def clean(self, text):
		words = self.tokenize(text)
		words = self.remove_non_ascii(words)
		words = self.remove_punctuation2(words)
		words = self.replace_numbers(words)
		return ' '.join(words)
		# return words

	def get_modified_data(self, FILE_PATH):
		f = open(FILE_PATH, 'r')
		data_processed = []
		for line in f.readlines():
			line = line.strip()
			temp = line.split('\t')
			for i in range(2):
				temp[i] = self.clean(temp[i])
			data_processed.append(temp)
		f.close()
		return data_processed

	def clean_str(self, string):
		"""
		Tokenization/string cleaning for all datasets except for SST.
		Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
		"""
		string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		string = re.sub(r"\'s", " \'s", string)
		string = re.sub(r"\'ve", " \'ve", string)
		string = re.sub(r"n\'t", " n\'t", string)
		string = re.sub(r"\'re", " \'re", string)
		string = re.sub(r"\'d", " \'d", string)
		string = re.sub(r"\'ll", " \'ll", string)
		string = re.sub(r",", " , ", string)
		string = re.sub(r"!", " ! ", string)
		string = re.sub(r"\(", " \( ", string)
		string = re.sub(r"\)", " \) ", string)
		string = re.sub(r"\?", " \? ", string)
		string = re.sub(r"\s{2,}", " ", string)
		return string.strip().lower()


	def load_data_and_labels(self, train_file):
		"""
		Loads MR polarity data from files, splits the data into words and generates labels.
		Returns split sentences and labels.
		"""
		# Load data from files
		train_data = list(open(train_file, "r").readlines())
		questions = [s.split('\t')[0].strip() for s in train_data]
		answers = [s.split('\t')[1].strip() for s in train_data]
		# Split by words
		x_text = questions + answers
		# print(x_text)
		x_text = [self.clean_str(sent) for sent in x_text]
		# Generate labels
		# positive_labels = [[0, 1] for _ in positive_examples]
		# negative_labels = [[1, 0] for _ in negative_examples]
		# y = np.concatenate([positive_labels, negative_labels], 0)
		# return [x_text, y]
		return x_text

	
	def main(self):
		data = self.get_modified_data(FILE_PATH)
		# print(type(data[1][1]))
		# a = []
		# b = []
		# for i in range(10):
		# 	temp = []
		# 	temp.extend([data[i][0]])
		# 	temp.extend([data[i][1]])
		# 	a.append(temp)
		# 	b.extend([data[i][2]])
		# print (a[1])
		print (data)

if __name__ == "__main__":
	a = PreprocessData()
	a.main()


		



