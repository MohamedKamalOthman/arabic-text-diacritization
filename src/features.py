from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# data = ["السلام عليكم", "على النحو الذي يمكن إدراكه مع الجملة البسيطة، من المهم معرفة المكان الذي تبدأ أو تنتهي عنده الرسالة الكلمة، للفهم بسهولة ويسر"]


def bag_of_words(data: list[str]):
    vectorizer = CountVectorizer(analyzer="char")
    bow = vectorizer.fit_transform(data)
    vocabulary = vectorizer.get_feature_names_out()
    bow_array = bow.toarray()

    # Normalizing by the sum of each row
    bow_array = bow_array / bow_array.sum(axis=1, keepdims=True)

    return vocabulary, bow_array


def tf_idf(data: list[str]):
    tfidf_vectorizer = TfidfVectorizer(analyzer="char")
    tfidf = tfidf_vectorizer.fit_transform(data)
    tfidf_vocab = tfidf_vectorizer.get_feature_names_out()
    return tfidf_vocab, tfidf.toarray()
