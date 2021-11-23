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

total_correct = 0
total_answered = 0
synonyms_csv = open('../data/synonyms.csv', mode='r')

with open('word2vec-google-news-300-details.csv', 'w', newline='') as output_file:
    output_writer = csv.writer(output_file)
    parsed_file = csv.reader(synonyms_csv, delimiter=',')
    for line in parsed_file:
        question = line[0]
        answer = line[1]

        label = None
        # (word, score) => 81 instances in test file
        prediction = [None, 0]
        try:
            for i in range(2, 6):
                score = wv.similarity(question, line[i])
                if score > prediction[1]:
                    prediction = [line[i], score]
            if prediction[0] == answer:
                total_correct += 1
                label = 'correct'
            else:
                label = 'wrong'
            total_answered += 1
        except:
            label = "guess"

            # Guessing first when word not found
            prediction[0] = line[2]

        row = [question, answer, prediction[0], label]
        output_writer.writerow(row)

# Task 1.2

with open('analysis.csv', 'w', newline='') as analysis_file:
    analysis_writer = csv.writer(analysis_file)

    vocab = wv.index_to_key
    accuracy = total_correct/total_answered

    row = []
    row.append("word2vec-google-news-300")

    row.append(len(vocab))
    row.append(total_correct)
    row.append(total_answered)
    row.append(accuracy)
    analysis_writer.writerow(row)


synonyms_csv.close()
output_file.close()
analysis_file.close()
