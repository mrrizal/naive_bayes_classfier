import operator
from string import punctuation
from pprint import pprint


class NaiveBayes:
    def __init__(self):
        self.translator = str.maketrans('', '', punctuation)

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
        # tokenizer dan remove punctuation
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

    def tokenize(self, sentence):
        sentence = sentence.lower().split()
        return sentence

    def train(self, dataset):
        self.prob_kelas(dataset)
        dataset = self.preprocess_dataset(dataset)
        self.word_to_vector(dataset)
        self.hit_likelihood()

    def predict(self, sentence):
        words = self.tokenize(sentence)
        hasil = {}
        for i in self.classes:
            result = self.prior[i]
            for j in words:
                if j in self.likelihood:
                    result *= self.likelihood[j][i]
                else:
                    result *= 1 / (self.n_words[i] + len(self.vocabulary) + 1)

            hasil[i] = result

        hasil['label'] = max(hasil.items(), key=operator.itemgetter(1))[0]
        hasil['sentence'] = sentence
        return hasil


if __name__ == '__main__':
    naivebayes = NaiveBayes()
    dataset = naivebayes.open_dataset('tweets_train.txt')
    naivebayes.train(dataset)
    result = naivebayes.predict('damn, i hate you')
    pprint(result)