from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    tira = Client()

    # loading train data
    text_train = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    targets_train = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )
    ###########################################################
    np_train = text_train.to_numpy()
    np_targets = targets_train.to_numpy()
    merged_data = []
    for element in np_train:
        id_text = element[0]
        text = element[1]
        boolean_row = np_targets[np.where(np_targets[:,0] == id_text)]
        boolean = boolean_row[0][1] if boolean_row.size > 0 else None
        merged_data.append([id_text,text,boolean])

    bow_list = []
    for element in np_train:
        temp = element[1].split()
        bow_list.extend(temp)
    vectorizer = CountVectorizer()
    vectorizer.fit(bow_list)
    amount_dict = vectorizer.vocabulary_
    sorted_words = sorted(amount_dict.items(), key=lambda x: x[1], reverse=True)
    word_to_id = {word: idx for idx, (word, _) in enumerate(sorted_words)}
    vector_dict = {}
    for element in merged_data:
        temp = [x.lower() for x in element[1].split()]
        temp_list = []
        for word in temp:
            if word in word_to_id:
                temp_list.append(word_to_id[word])
            else:
                temp_list.append(len(word_to_id)) 
        vector_dict[tuple(temp_list)] = element[2] 
    print(vector_dict)
    
    X = np.array(list(vector_dict.keys())) 
    y = np.array([vector_dict[key] for key in vector_dict]) 
    X = np.array([vector + [0] * (max_len - len(vector)) for vector in X])
    max_len = max(len(vector) for vector in X)

    model = LogisticRegression()
    model.fit(X, y)

    y_pred_train = model.predict(X)
    accuracy_train = accuracy_score(y, y_pred_train)
    print("Trainingsgenauigkeit:", accuracy_train)
    ###############################################################################
    # classifying the data
    prediction = (
        text_validation.set_index("id")["text"]
        .str.contains("delve", case=False)
        .astype(int)
    )

    # converting the prediction to the required format
    prediction.name = "generated"
    prediction = prediction.reset_index()

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
