!pip install JPype1
import streamlit as st
import tensorflow as tf
import numpy as np
import jpype
from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java
from typing import List
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#  ----------------Metin Önişleme-----------------
stopwords_turkish = set(stopwords.words('turkish'))
stopwords_turkish.update(["bir", "kadar", "sonra", "kere", "mi", "ye", "te", "ta", "nun", "daki", "nın", "ten"])
silinecek_tokenler = ['ol', 'et', 'yap', 'ben', 'ver', 'al', 'de', 'gel', 'yıl', 'gün', 'kendi', 'çık', 'söyle', 'ara', 'iş', 'var', 'son', 'yer', 'gör', 'git', 'başla', 'ilk', 'bulun', 'başkan', 'konu', 'türkiye', 'kullan', 'yüz', 'çalış']
stopwords_turkish.update(silinecek_tokenler)

url_pattern = re.compile(r'https?://\S+')

def remove_urls(text):
    return url_pattern.sub('', text)

def remove_stopwords(text):
    return " ".join(word for word in text.split() if word.lower() not in stopwords_turkish)

def preprocessing(text):
    text = text.lower()

    text = remove_urls(text)

    text = re.sub(r'([.,])\s*', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\d', '', text)

    text = re.sub(r'\n', ' ', text)

    text = re.sub(r'<.*?>', '', text)

    text = remove_stopwords(text)

    return text

#-----------------------Zemberek-------------------------

class Zemberek:
    def __init__(self):
        self.start_jvm()

    def start_jvm(self):
        if not jpype.isJVMStarted():
            jpype.startJVM(
                getDefaultJVMPath(),
                "-ea",
                "-Djava.class.path=/content/zemberek-full.jar",
                convertStrings=False,
            )

        TurkishMorphology = JClass("zemberek.morphology.TurkishMorphology")
        self.morphology = TurkishMorphology.createWithDefaults()

    def stop_jvm(self):
        shutdownJVM()

    def stem_word(self, word: str) -> str:
        result = self.morphology.analyzeAndDisambiguate(word).bestAnalysis()
        stems = [analysis.getStem() for analysis in result]
        return stems[0] if stems else ""

    def lemmatize_word(self, word: str) -> str:
        result = self.morphology.analyzeAndDisambiguate(JString(word)).bestAnalysis()
        lemmas = []
        for i in range(result.size()):
            lemma = result.get(i).getLemmas().toString() if result.get(i).getLemmas().size() > 0 else ""
            lemmas.append(str(lemma))
        return lemmas[0] if lemmas else ""



    def stem_sentence(self, sentence: str) -> List[str]:
        return [self.stem_word(word) for word in sentence.split()]

    def lemmatize_sentence(self, sentence: str) -> List[str]:
        return [self.lemmatize_word(word) for word in sentence.split()]






zemberek = Zemberek()


# Streamlit uygulama ayarları
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('/content/nlp_model.hdf5')
    return model

model = load_model()

tokenizer = Tokenizer()
max_length = 500  # Maksimum sequence uzunluğu
padding_type='post'
truncating_type='post'

class_names = ["bilim-teknoloji", "finans-ekonomi", "kültür-sanat", "magazin", "sağlık", "siyaset", "spor", "turizm", "çevre"]

# Streamlit uygulaması

st.write("""
         # Haber Metni Sınıflandırma
         """)

# Kullanıcıdan doğrudan metin girmesini isteyin
text_input = st.text_area("Lütfen bir haber metni girin")

if st.button("Tahmin Et"):
    if text_input:
        st.write("Girilen metin:")
        st.write(text_input)

        # Ön işleme
        processed_text = preprocessing(text_input)
        st.write("İşlenmiş metin:")
        st.write(processed_text)

        # Zemberek ile lemmatization
        st.write("Kökenler (Lemmatization):")
        lemmatized_sentence = zemberek.lemmatize_sentence(processed_text)
        lemmatized_text = " ".join(lemmatized_sentence)
        st.write(lemmatized_text)


        # Tokenize ve Pad İşlemleri
        tokenizer.fit_on_texts([lemmatized_text])
        sequences = tokenizer.texts_to_sequences([lemmatized_text])
        st.write("Tokenized Sequences:")
        st.write(sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=truncating_type)

        # Tahmin yap
        prediction = model.predict(padded_sequences)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]

        st.write("Tahmin edilen sınıf:", predicted_class_name)
        st.write("Tahmin olasılıkları:", prediction)