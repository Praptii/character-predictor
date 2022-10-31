import collections
import numpy as np

class Unigram(object):
    """Barebones example of a language model class."""

    def __init__(self):
        self.counts = collections.Counter()
        self.total_count = 0        

    def train(self, filename):
        """Train the model on a text file."""
        for line in open(filename):            
            for w in line.rstrip('\n'):
                self.counts[w] += 1
                self.total_count += 1

    def start(self):
        """Reset the state to the initial state."""
        self.state = None       

    def read(self, w):
        """Read in character w, updating the state."""              
        self.state = w
    
    def prob(self, w):
        """Return the probability of the next character being w given the
        current state."""
        return self.counts[w] / self.total_count

    def predict(self):
        maximum = 0
        word = None
        
        for w in self.counts.keys():            
            if(self.prob(w)) > maximum:                
                maximum = self.prob(w)
                word = w                 
        return word
     
    
def accuracy(model, test_set):
    corpus_test = ''
           
    for line in open(test_set):                   
        for w in line.rstrip('\n'):
            corpus_test = corpus_test + w        
   
    correct = 0
    
    model.start()
    
    for index in range(0, len(corpus_test)):            
            model.read(corpus_test[index])
            if model.predict() == corpus_test[index]:                
                correct += 1
    
    return correct/len(corpus_test) 