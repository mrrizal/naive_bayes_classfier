import operator
import math
from random import shuffle
from functools import reduce
from string import punctuation
from pprint import pprint


class NaiveBayes:
    def open_dataset(self, path):
        data = [i for i in open(path).read().split('\n') if i != '']
        return data

    def prob_kelas(self, dataset):
        # hitung probabilitas tiap kelas
        n = len(dataset)
        self.prior = {}
        for i in dataset:
            try:
                data = i.split('\t')
                if data[1] not in self.prior:
                    self.prior[data[1]] = 1
                else:
                    self.prior[data[1]] += 1
            except IndexError:
                continue

        for i in self.prior:
            self.prior[i] = self.prior[i] / n

        self.classes = list(self.prior.keys())

    def preprocess_dataset(self, dataset):
        # tokenizer
        result = []
        for i in dataset:
            data = i.split('\t')
            data[0] = data[0].lower().split()
            result.append(data)

        return result

    def word_to_vector(self, dataset):
        self.words = {}
        self.n_words = dict([(i, 0) for i in self.classes])
        for i in dataset:
            for j in i[0]:
                try:
                    if j not in self.words:
                        self.words[j] = dict(
                            (kelas, 0) for kelas in self.classes)
                        self.words[j][i[1]] += 1
                    else:
                        self.words[j][i[1]] += 1
                except IndexError:
                    continue

        self.vocabulary = list(self.words.keys())

        for i in self.n_words.keys():
            for j in self.words.values():
                self.n_words[i] += j[i]

    def hit_likelihood(self):
        self.likelihood = {}
        for i in self.words.items():
            key = i[0]
            temp = {}
            for j in i[1].items():
                temp[j[0]] = (j[1] + 1) / (
                    self.n_words[j[0]] + len(self.vocabulary))
            self.likelihood[key] = temp

    def train(self, dataset):
        self.prob_kelas(dataset)
        dataset = self.preprocess_dataset(dataset)
        self.word_to_vector(dataset)
        self.hit_likelihood()

    def split_dataset(self, dataset):
        hitung = int(80 / 100 * len(dataset))
        return dataset[0:hitung], dataset[hitung:]

    def predict(self, sentence):
        words = sentence.lower().split()
        hasil = {}
        for i in self.classes:
            result = self.prior[i]
            temp = []
            for j in words:
                if j in self.likelihood:
                    temp.append(math.log(self.likelihood[j][i]))
                else:
                    temp.append(
                        math.log(1 /
                                 (self.n_words[i] + len(self.vocabulary) + 1)))

            hasil[i] = math.log(result) + reduce(lambda x, y: x + y, temp)

        hasil['label'] = max(hasil.items(), key=operator.itemgetter(1))[0]
        hasil['sentence'] = sentence
        return hasil

    def hitung_akurasi(self, datatest):
        counter = 0
        for i in datatest:
            temp = i.split('\t')
            if len(temp) == 2:
                result = self.predict(temp[0])
                if result['label'] == temp[1]:
                    counter += 1

        return (counter / len(datatest)) * 100


if __name__ == '__main__':
    naivebayes = NaiveBayes()
    dataset = naivebayes.open_dataset('tweets_train.txt')
    shuffle(dataset)
    datatrain, datatest = naivebayes.split_dataset(dataset)
    naivebayes.train(datatrain)
    akurasi = naivebayes.hitung_akurasi(datatest)
    print("akurasi : %s %%" % str(akurasi))
    print()

    mystring = [
        "Dear Palestine , our thoughts , prayers and loves are always with you :') #SavePalestine #SaveAlAqsa",
        "Jerusalem isn't the capital of Israel!! Fuck you the fucking American!! #SavePalestine #KamiBersamaPalestina #doakamiuntukpalestina",
        "to all my muslim family. may Allah always bless wherever you are . #SavePalestine #pleasekeeppraying",
        "May allah punish you for what you’ve done trump #SavePalestine",
        "You do wrong thing Trump , you will get bad consequence !! #savepalestine #jerrussalamiscapitalpalestine"
    ]

    for i in mystring:
        pprint(naivebayes.predict(i))
        print()
        print()