import csv
import gensim.downloader as api


def generate_details_and_analysis(corpos):
    wv = api.load(corpos)

    total_correct = 0
    total_answered = 0
    synonyms_csv = open('../data/synonyms.csv', mode='r')

    with open(f'{corpos}-details.csv', 'a', newline='') as output_file:
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

    with open('analysis.csv', 'a', newline='') as analysis_file:
        analysis_writer = csv.writer(analysis_file)

        vocab = wv.index_to_key
        accuracy = total_correct / total_answered if total_answered != 0 else total_answered

        row = []
        row.append(corpos)
        row.append(len(vocab))
        row.append(total_correct)
        row.append(total_answered)
        row.append(accuracy)
        analysis_writer.writerow(row)

    synonyms_csv.close()
    output_file.close()
    analysis_file.close()


corporas = ['fasttext-wiki-news-subwords-300',
            'glove-twitter-25',
            'glove-twitter-50',
            'glove-wiki-gigaword-300']

for corpora in corporas:
    generate_details_and_analysis(corpora)
