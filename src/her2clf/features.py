from sklearn.preprocessing import OneHotEncoder

def fit_onehot_encoder(sequences):
    seq_lists = sequences.apply(list).tolist()
    encoder=OneHotEncoder(handle_unknown="ignore")
    X=encoder.fit_transform(seq_lists)

    return X, encoder

def transform_sequences(sequences, encoder):
    seq_lists = sequences.apply(list).tolist()
    X = encoder.transform(seq_lists)

    return X