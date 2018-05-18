import json

def create_csv(input_file, output_file):

    label_kinds = ["FOOD#QUALITY", "RESTAURANT#GENERAL", "SERVICE#GENERAL", "AMBIENCE#GENERAL",
                   "RESTAURANT#MISCELLANEOUS", "FOOD#PRICES", "RESTAURANT#PRICES", "DRINKS#QUALITY",
                   "LOCATION#GENERAL", "DRINKS#PRICES"]

    sentences = []
    labels = []

    with open(input_file) as j:
        reviews = json.load(j)['Reviews']['Review']
        for d in reviews:
            for s in d["sentences"]['sentence']:
                # register words
                if "@OutOfScope" not in s and "Opinions" in s and "text" in s:
                    if isinstance(s["Opinions"]["Opinion"], list):
                        annotations = [o["@category"] for o in s["Opinions"]["Opinion"]]
                    else:
                        annotations = [s["Opinions"]["Opinion"]["@category"]]
                if len(annotations) > 0:
                    try:
                        row = []
                        for k in label_kinds:
                            if k in annotations:
                                row.append('1')
                            else:
                                row.append('0')
                        labels.append(row)
                        sentences.append(s['text'])
                    except TypeError:
                        continue

    with open(output_file, 'w') as file:
        file.write('text_' + "_".join(label_kinds) + "\n")
        for i in range(len(sentences)):
            line = sentences[i] + str('_') + "_".join(labels[i]) + "\n"
            try:
                file.write(line)
            except UnicodeEncodeError:
                continue

create_csv("Data/ABSA-15_Restaurants_Train.json", 'Data/aspects_train.csv');
create_csv("Data/ABSA15_Restaurants_Test.json", 'Data/aspects_test.csv');