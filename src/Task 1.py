import csv
import gensim.downloader as api
from gensim.models import Word2Vec

wv = api.load('word2vec-google-news-300')

for index, word in enumerate(wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(wv.index_to_key)} is {word}")

#  cosine similarity between pairs of embeddings
pairs = [
    ('car', 'minivan'),  # a minivan is a kind of car
    ('car', 'bicycle'),  # still a wheeled vehicle
    ('car', 'airplane'),  # ok, no wheels, but still a vehicle
    ('car', 'cereal'),  # ... and so on
    ('car', 'communism'),
]

for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))

# Task 1.1

synonyms_csv = open('../data/synonyms.csv', mode='r')

with open('task1.1_output.csv', 'w', newline='') as output_file:
    output_writer = csv.writer(output_file)
    parsed_file = csv.reader(synonyms_csv, delimiter=',')
    for line in parsed_file:
        question = line[0]
        answer = line[1]

        # (word, score)
        label = None
        prediction = [None, 0]
        try:
            for i in range(2, 6):
                score = wv.similarity(question, line[i])
                if score > prediction[1]:
                    prediction = [line[i], score]
            if prediction[0] == answer:
                label = 'correct'
            else:
                label = 'wrong'
        except:
            label = "guess"

            # Guessing first when word not found
            prediction[0] = line[2]

        row = [question, answer, prediction[0], label]
        output_writer.writerow(row)

synonyms_csv.close()
output_file.close()

