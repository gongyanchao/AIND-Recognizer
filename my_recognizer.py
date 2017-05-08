import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    #raise NotImplementedError
    for i in test_set._hmm_data.keys():
        X, lengths = test_set._hmm_data[i]
        dict_p = {}
        #print("Predicting the %s word" % i)
        for word in models.keys():
            model = models[word]
            #print("Predicting prob of word %s" % word)
            try:
                logL = model.score(X, lengths)
            except:
                #print("failed to score, mark as -inf")
                logL = float('-inf')
            dict_p[word] = logL
        probabilities.append(dict_p)
    #print(probabilities)
    for prob in probabilities:
        ordered =  sorted(prob.items(), key =  lambda x:x[1], reverse=True)
        #print(ordered)
        guess = ordered[0][0]
        guesses.append(guess)

    return probabilities, guesses

