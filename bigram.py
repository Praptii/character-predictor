chinese_pronc_dict = dict()

for line in open('chinese/charmap', encoding = 'utf-8'):            
    chars = tuple(line.strip().split(" "))

    if chars[1] in chinese_pronc_dict:
        chinese_pronc_dict[chars[1]].append(chars[0])
    else:
        chinese_pronc_dict[chars[1]] = [chars[0]]


class Bigram():
    def __init__(self, smoothing = True, alpha = 1):
        self.smoothing = True
        self.alpha = 1
        self.bi_gram_counts = collections.Counter()
        self.state = list()
        self.characters = []

    def obtain_bi_grams(self, sentence):
        characters = list(sentence)

        bi_grams = list(zip(*[characters[i:] for i in range(0,2)]))
        return bi_grams

    def train(self, train_file):
         for line in open(train_file, encoding = 'utf-8'):

            for c in line.rstrip('\n'):
                if c not in self.characters:
                    self.characters.append(c)

            bi_grams = self.obtain_bi_grams(line.strip())
            for gram in bi_grams:
                self.bi_gram_counts[gram] += 1

    def start(self):
        """Reset the state to the initial state."""
        self.state = list()

    def read(self, w):
        """Read in character w, updating the state."""
        if len(self.state) < 1:
            self.state.append(w)
        else:
            self.state.pop(0)
            self.state.append(w)

    def prob(self, c):
        """Return the probability of the next character being w given the
        current state."""

        total_count = 0

        # For every character
        for character in self.characters:
            total_count += self.bi_gram_counts[(self.state[0],character)]

        return (self.bi_gram_counts[(self.state[0],c)]  + self.alpha) / (total_count + len(self.characters))

    def predict(self, token):
        tokens = candidates(token)
        maximum = 0
        char = None

        for token in tokens:
            if(self.prob(token)) > maximum:
                maximum = self.prob(token)
                char = token
        return char

    def accuracy(self,  test_file_p, test_file_h):
        file_p = []
        correct_ones = 0
        for line in open(test_file_p, encoding = 'utf-8'):
            file_p.append(line.strip().split(" "))

        file_h = []
        for line in open(test_file_h, encoding = 'utf-8'):
            file_h.append(list(line.strip()))

        predicted_chars = []
        actual_chars = []

        for line_p, line_h in zip(file_p, file_h):
            for index in range(1, len(line_h)):
                self.read(line_h[index - 1])

                actual_chars.append(line_h[index])
                predicted_chars.append(self.predict(line_p[index]))
        total = len(actual_chars)
        for a,p in zip(actual_chars, predicted_chars):
            if a == p:
                correct_ones += 1
        return correct_ones/total
