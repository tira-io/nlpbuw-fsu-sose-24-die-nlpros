from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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
    max_length = 0
    for element in np_train:
        temp = element[1].split()
        if max_length < len(temp):
            max_length = len(temp)
        bow_list.extend(temp)
    print("max length " + str(max_length))
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
    #print(vector_dict)
    np_test = text_validation.to_numpy()
    np_ttargets = targets_validation.to_numpy()
    merged_test = []
    for element in np_test:
        id_text = element[0]
        text = element[1]
        boolean_row = np_ttargets[np.where(np_ttargets[:,0] == id_text)]
        boolean = boolean_row[0][1] if boolean_row.size > 0 else None
        merged_test.append([id_text,text,boolean])

    test_dict = {}
    for element in merged_test:
        temp = [x.lower() for x in element[1].split()]
        temp_list = []
        for word in temp:
            if word in word_to_id:
                temp_list.append(word_to_id[word])
            else:
                temp_list.append(len(word_to_id)) 
        test_dict[tuple(temp_list)] = element[2]
    #print(test_dict) 

    def pad_vectors(vectors, max_length, padding_value=-1):
        padded_vectors = []
        for vector in vectors:
            if len(vector) < max_length:
                padded_vector = list(vector)
                padded_vector.extend([padding_value]*(max_length - len(vector)))
            else:
                padded_vector = vector[:max_length]  # Truncate if longer than max_length
            padded_vectors.append(padded_vector)
        return padded_vectors

    X = list(vector_dict.keys())
    X = pad_vectors(X, max_length, padding_value=-1)
    y = np.array([vector_dict[key] for key in vector_dict])
    X = np.array(X)
    X_test = list(test_dict.keys())
    X_test = pad_vectors(X_test, max_length, padding_value=-1)
    X_test = np.array(X_test)
    y_test = np.array([test_dict[key] for key in test_dict])
    print(X.shape)
    print(y.shape)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.fit_transform(X_test)
    model = LogisticRegression(max_iter = 2000)
    model.fit(X_scaled, y)

    y_pred_train = model.predict(X_scaled)
    y_pred_test = model.predict(X_test)
    print(y_pred_train)
    print(y_pred_test)
    print(len(y_pred_test))
    accuracy_train = accuracy_score(y, y_pred_train)
    accuracy_test = accuracy_score(y_test,y_pred_test)
    print("Trainingsgenauigkeit:", accuracy_train)
    print("Testgenauigkeit: ", accuracy_test)

    


    # classifying the data
    prediction = (
        text_validation.set_index("id")["text"]
        .str.contains("delve", case=False)
        .astype(int)
    )
    print(prediction)
    # converting the prediction to the required format
    prediction.name = "generated"
    prediction = prediction.reset_index()
    print(prediction)
    

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
