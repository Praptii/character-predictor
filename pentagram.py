class Pentagram():
    def __init__(self, smoothing = True, alpha = 1):
        self.smoothing = True
        self.alpha = 1
        self.five_gram_counts = collections.Counter()
        self.state = list()
        self.characters = []
        
    def obtain_5_grams(self, sentence):
        characters = list(sentence)       
    
        # https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/
        five_grams = list(zip(*[characters[i:] for i in range(0,5)]))
        return five_grams
        
    def train(self, train_file):
         for line in open(train_file):   
                
            for c in line.rstrip('\n'):
                if c not in self.characters:
                    self.characters.append(c)
                
            five_grams = self.obtain_5_grams(line.strip())               
            for gram in five_grams:
                self.five_gram_counts[gram] += 1
    
    def start(self):
        """Reset the state to the initial state."""
        self.state = list()       

    def read(self, w):
        """Read in character w, updating the state."""              
        if len(self.state) < 4:
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
            total_count += self.five_gram_counts[(self.state[0],self.state[1],self.state[2],self.state[3],character)]
                    
        return (self.five_gram_counts[(self.state[0],self.state[1],self.state[2],self.state[3],c)]  
                + self.alpha) / (total_count + len(self.characters))
    
    def predict(self):
        maximum = -100000000
        char = None
        
        for c in self.characters:            
            if(np.log(self.prob(c))) > maximum:                
                maximum = np.log(self.prob(c))
                char = c                 
        return char
    
    def accuracy(model, test_set):
        corpus_test = ''

        for line in open(test_set):                   
            for w in line.rstrip('\n'):
                corpus_test = corpus_test + w        

        correct = 0
        model.start()

        for index in range(0, len(corpus_test) - 1):            
            model.read(corpus_test[index])     

            if index > 2:
    #             print('Actual', corpus_test[index],'Predicted', model.predict())
                if model.predict() == corpus_test[index + 1]:                
                    correct += 1   

        return correct/(len(corpus_test) - 4)