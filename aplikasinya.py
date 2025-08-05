import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer # Import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # Import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory # Import StopWordRemoverFactory

# Set page config harus menjadi perintah Streamlit pertama
st.set_page_config(layout="wide")

# Pastikan NLTK punkt_tab diunduh (stopwords tidak lagi dari NLTK)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Inisialisasi stemmer di luar fungsi agar hanya dibuat sekali
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Inisialisasi StopWordRemover di luar fungsi agar hanya dibuat sekali
stopword_factory = StopWordRemoverFactory()
list_stopwords = set(stopword_factory.get_stop_words())

# --- Fungsi Preprocessing (dari notebook Preprocessing.ipynb) ---
def clean_youtube_text(text):
    """
    Membersihkan teks dari karakter yang tidak diinginkan, tautan, hashtag,
    dan mention, serta mengubahnya menjadi huruf kecil.
    """
    text = str(text) # Memastikan input adalah string
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Menghapus mention
    text = re.sub(r'#\w+', '', text)  # Menghapus hashtag
    text = re.sub(r'https?://\S+', '', text)  # Menghapus tautan
    text = re.sub(r'[^A-Za-z0-9 ]', '', text)  # Menghapus karakter non-alfanumerik
    text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebih
    return text.lower() # Mengubah ke huruf kecil

# --- Fungsi Normalisasi (dari notebook Preprocessing.ipynb) ---
kamus_normalisasi = {}
try:
    kamus = pd.read_excel('kamus_normalisasi_lengkap.xlsx')
    kamus_normalisasi = dict(zip(kamus['slang'], kamus['formal']))
except FileNotFoundError:
    st.error("File 'kamus_normalisasi_lengkap.xlsx' tidak ditemukan. Pastikan file ini berada di direktori yang sama dengan aplikasi Streamlit.")
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat 'kamus_normalisasi_lengkap.xlsx': {e}")

def normalisasi_text(text, kamus_normalisasi):
    """
    Normalisasi kata-kata dalam teks menggunakan kamus normalisasi.
    """
    kata_kata = text.split()
    normalisasi_kata_kata = [kamus_normalisasi.get(kata, kata) for kata in kata_kata]
    return ' '.join(normalisasi_kata_kata)

# --- Fungsi Stopword Removal (dari notebook Preprocessing.ipynb, disesuaikan) ---
def stopword_removal(words):
    """
    Menghapus stopword dari daftar kata.
    """
    return [word for word in words if word not in list_stopwords]

# --- Fungsi Tokenisasi (dari notebook Preprocessing.ipynb) ---
def tokenize_text(text):
    """
    Melakukan tokenisasi teks menjadi daftar kata.
    """
    return word_tokenize(text)

# --- Fungsi Stemming (dari notebook Preprocessing.ipynb) ---
def apply_stemming(tokens):
    """
    Menerapkan stemming pada daftar token.
    """
    return [stemmer.stem(word) for word in tokens]

# --- Fungsi Preprocessing Lengkap (untuk konsistensi) ---
def full_preprocessing(text, kamus_normalisasi, list_stopwords, stemmer):
    """
    Menerapkan seluruh alur preprocessing: cleaning, normalisasi, tokenisasi,
    stopword removal, dan stemming.
    """
    cleaned_text = clean_youtube_text(text)
    normalized_text = normalisasi_text(cleaned_text, kamus_normalisasi)
    tokenized_text = tokenize_text(normalized_text)
    filtered_text = stopword_removal(tokenized_text)
    stemmed_text_tokens = apply_stemming(filtered_text) # APPLY STEMMING HERE
    return ' '.join(stemmed_text_tokens)

# --- Load Model dan Vectorizer ---
@st.cache_resource # Cache the model and vectorizer to avoid reloading on every rerun
def load_model_and_vectorizer():
    best_model = None
    label_encoder = None
    tfidf_vectorizer = None
    X_train_smote = None
    y_train_smote = None
    X_test = None
    y_test = None

    # Coba memuat model_terbaik.pkl terlebih dahulu (berisi model, label_encoder, tfidf_vectorizer)
    try:
        model_data = joblib.load('model_terbaik.pkl')
        best_model = model_data.get('model')
        label_encoder = model_data.get('label_encoder')
        tfidf_vectorizer = model_data.get('tfidf_vectorizer')

        if best_model is None or label_encoder is None or tfidf_vectorizer is None:
            st.error("File 'model_terbaik.pkl' tidak berisi semua komponen yang diharapkan (model, label_encoder, tfidf_vectorizer).")
            return None, None, None, None, None, None, None

    except FileNotFoundError:
        st.error("File 'model_terbaik.pkl' tidak ditemukan. Pastikan file ini ada di direktori yang sama.")
        return None, None, None, None, None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat 'model_terbaik.pkl': {e}")
        return None, None, None, None, None, None, None

    # Coba memuat penyesuaian_data.pkl untuk data split train/test
    try:
        data_loaded_for_splits = joblib.load('penyesuaian_data.pkl')
        X_train_smote = data_loaded_for_splits.get('X_train_smote')
        y_train_smote = data_loaded_for_splits.get('y_train_smote')
        X_test = data_loaded_for_splits.get('X_test')
        y_test = data_loaded_for_splits.get('y_test')

        if X_train_smote is None or y_train_smote is None or X_test is None or y_test is None:
            st.warning("File 'penyesuaian_data.pkl' tidak berisi semua komponen split data yang diharapkan (X_train_smote, y_train_smote, X_test, y_test). Dashboard mungkin menampilkan informasi terbatas.")

    except FileNotFoundError:
        st.warning("File 'penyesuaian_data.pkl' tidak ditemukan. Informasi pembagian data dan performa model di Dashboard mungkin tidak tersedia.")
    except Exception as e:
        st.warning(f"Terjadi kesalahan saat memuat 'penyesuaian_data.pkl' untuk data split: {e}. Informasi pembagian data dan performa model di Dashboard mungkin tidak tersedia.")

    return tfidf_vectorizer, best_model, label_encoder, X_train_smote, y_train_smote, X_test, y_test

tfidf_vectorizer, best_model, label_encoder, X_train_smote, y_train_smote, X_test, y_test = load_model_and_vectorizer()

# --- Fungsi Prediksi Sentimen ---
def predict_sentiment(text):
    if tfidf_vectorizer is None or best_model is None or label_encoder is None:
        st.warning("Model atau vektorizer belum dimuat. Tidak dapat melakukan prediksi.")
        return "Error: Model not loaded"

    # Preprocessing teks input menggunakan full_preprocessing
    # Ini akan memastikan konsistensi dengan cara TF-IDF dilatih (setelah stemming)
    processed_text = full_preprocessing(text, kamus_normalisasi, list_stopwords, stemmer)

    # Transformasi teks menggunakan TF-IDF Vectorizer yang sudah dilatih
    text_vectorized = tfidf_vectorizer.transform([processed_text])

    # Prediksi sentimen
    prediction = best_model.predict(text_vectorized)
    sentiment = label_encoder.inverse_transform(prediction)[0]

    return sentiment

# --- Streamlit UI ---
st.title("ANALISIS SENTIMEN PENGGUNA YOUTUBE TERHADAP DAMPAK ARTIFICIAL INTELLIGENCE MENGGUNAKAN ALGORITMA NAIVE BAYES")

if 'menu_selection' not in st.session_state:
    st.session_state.menu_selection = "Dashboard"

st.markdown("""
<style>
    .stSidebar .stButton > button {
        background-color: transparent !important;
        color: #000000 !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 0.25rem !important;
        width: 100%;
        text-align: left !important;
        margin-bottom: 0.5rem;
        justify-content: flex-start !important;
    }

    .stSidebar .stButton > button:hover {
        background-color: rgba(0, 123, 255, 0.1) !important;
        color: #000000 !important;
    }

    .stSidebar .stButton > button:active {
        background-color: rgba(0, 123, 255, 0.2) !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigasi")

if st.sidebar.button("Dashboard", key="nav_dashboard"):
    st.session_state.menu_selection = "Dashboard"
if st.sidebar.button("Prediksi Sentimen", key="nav_predict"):
    st.session_state.menu_selection = "Prediksi Sentimen"

st.sidebar.markdown("---")

menu_selection = st.session_state.menu_selection

if menu_selection == "Dashboard":
    st.header("Dashboard Analisis Sentimen")

    if tfidf_vectorizer is not None and best_model is not None and label_encoder is not None:
        st.subheader("Ringkasan Data Komentar")
        try:
            df_labelled = pd.read_excel('data_labelled.xlsx')
            total_comments = len(df_labelled)
            st.write(f"Jumlah Total Komentar: **{total_comments}**")

            sentiment_counts = df_labelled['sentiment'].value_counts()
            sentiment_percentages = (sentiment_counts / total_comments * 100).round(2)

            st.write("Distribusi Sentimen:")
            for sentiment, count in sentiment_counts.items():
                st.write(f"- **{sentiment.capitalize()}**: {count} ({sentiment_percentages[sentiment]:.2f}%)")

            st.subheader("Visualisasi Distribusi Sentimen")
            sentiment_counts_df = sentiment_counts.reset_index()
            sentiment_counts_df.columns = ['Sentimen', 'Jumlah']

            fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Sentimen', y='Jumlah', data=sentiment_counts_df, ax=ax_dist, palette='viridis')
            ax_dist.set_title('Distribusi Sentimen Komentar')
            ax_dist.set_xlabel('Sentimen')
            ax_dist.set_ylabel('Jumlah Komentar')
            st.pyplot(fig_dist)

        except FileNotFoundError:
            st.warning("File 'data_labelled.xlsx' tidak ditemukan. Ringkasan data dan visualisasi distribusi sentimen mungkin tidak tersedia.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat atau memproses data untuk ringkasan: {e}")

        st.subheader("Informasi Pembagian Data (Train-Test Split)")
        if X_train_smote is not None and X_test is not None:
            st.write(f"Ukuran Data Pelatihan (setelah SMOTE): **{X_train_smote.shape[0]}** sampel")
            st.write(f"Ukuran Data Pengujian: **{X_test.shape[0]}** sampel")
        else:
            st.warning("Informasi pembagian data tidak tersedia. Pastikan 'penyesuaian_data.pkl' dimuat dengan benar dan berisi data split.")

        st.subheader("Performa Model Terbaik")
        if best_model is not None and X_test is not None and y_test is not None and label_encoder is not None:
            try:
                y_test_encoded = label_encoder.transform(y_test)
                y_pred = best_model.predict(X_test)

                accuracy = accuracy_score(y_test_encoded, y_pred)

                report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_, output_dict=True)

                macro_precision = report['macro avg']['precision']
                macro_recall = report['macro avg']['recall']
                macro_f1 = report['macro avg']['f1-score']

                st.write(f"Akurasi Model: **{accuracy:.4f}**")
                st.write(f"Precision (Macro Avg): **{macro_precision:.4f}**")
                st.write(f"Recall (Macro Avg): **{macro_recall:.4f}**")
                st.write(f"F1-Score (Macro Avg): **{macro_f1:.4f}**")

                st.write("Laporan Klasifikasi Lengkap:")
                report_df = pd.DataFrame(report).transpose().round(2)
                st.dataframe(report_df)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghitung performa model: {e}")
        else:
            st.warning("Performa model tidak dapat dihitung. Pastikan model dan data pengujian dimuat dengan benar dari file .pkl.")

        st.subheader("Alur Preprocessing Data")
        st.markdown("""
        Data komentar melalui beberapa tahapan preprocessing untuk memastikan kualitas dan relevansi data sebelum digunakan:
        1.  **Cleaning**: Menghapus karakter khusus, tautan (URL), hashtag, dan mention.
        2.  **Case Folding**: Mengubah teks menjadi huruf kecil
        3.  **Normalisasi**: Mengubah kata-kata tidak baku (slang) menjadi bentuk formal menggunakan kamus normalisasi.
        4.  **Tokenisasi**: Memecah teks menjadi unit-unit kata (token).
        5.  **Stopword Removal**: Menghapus kata-kata umum yang tidak memiliki makna signifikan (stopword) dalam Bahasa Indonesia.
        6.  **Stemming**: Mengubah kata-kata berimbuhan menjadi kata dasar.
        """)

        # Top 10 Kata Berdasarkan Total Skor TF-IDF
        st.subheader("Top 10 Kata Berdasarkan Total Skor TF-IDF")
        try:
            df_labelled = pd.read_excel('data_labelled.xlsx')
            # Preprocess comments for TF-IDF visualization using full_preprocessing
            # Ini akan memastikan konsistensi dengan cara TF-IDF dilatih (setelah stemming)
            df_labelled['processed_comments_for_tfidf'] = df_labelled['Comments'].apply(
                lambda x: full_preprocessing(x, kamus_normalisasi, list_stopwords, stemmer)
            )

            if not df_labelled.empty and tfidf_vectorizer is not None:
                # Filter hanya sentimen yang digunakan untuk pelatihan TF-IDF jika ada
                # Asumsi TF-IDF dilatih pada sentimen 'positif' dan 'negatif'
                df_model_filtered = df_labelled[df_labelled['sentiment'].isin(label_encoder.classes_)].copy()

                if not df_model_filtered.empty:
                    X_for_tfidf_viz = df_model_filtered['processed_comments_for_tfidf']
                    X_tfidf_viz = tfidf_vectorizer.transform(X_for_tfidf_viz)

                    feature_names = tfidf_vectorizer.get_feature_names_out()
                    tfidf_scores = X_tfidf_viz.sum(axis=0).A1

                    tfidf_df = pd.DataFrame({
                        'Kata Teratas': feature_names,
                        'Total_TFIDF': tfidf_scores
                    })

                    top_10_tfidf = tfidf_df.sort_values(by='Total_TFIDF', ascending=False).head(10).reset_index(drop=True)

                    st.dataframe(top_10_tfidf.style.format({'Total_TFIDF': '{:.14f}'}))

                else:
                    st.warning("Tidak cukup data berlabel 'positif' atau 'negatif' untuk visualisasi TF-IDF.")
            else:
                st.warning("Tidak cukup data berlabel atau TF-IDF Vectorizer tidak dimuat untuk visualisasi TF-IDF.")
        except FileNotFoundError:
            st.warning("File 'data_labelled.xlsx' tidak ditemukan. Visualisasi TF-IDF mungkin tidak tersedia.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat atau memproses data untuk visualisasi TF-IDF: {e}")


        # Visualisasi WordCloud untuk setiap sentimen
        st.subheader("WordCloud Sentimen")
        try:
            df_labelled = pd.read_excel('data_labelled.xlsx')
            # Preprocess comments for WordCloud (menggunakan full_preprocessing untuk konsistensi)
            df_labelled['processed_comments_for_wordcloud'] = df_labelled['Comments'].apply(
                lambda x: full_preprocessing(x, kamus_normalisasi, list_stopwords, stemmer)
            )

            text_positif = ' '.join(df_labelled[df_labelled['sentiment'] == 'positif']['processed_comments_for_wordcloud'].astype(str))
            text_negatif = ' '.join(df_labelled[df_labelled['sentiment'] == 'negatif']['processed_comments_for_wordcloud'].astype(str))
            text_netral = ' '.join(df_labelled[df_labelled['sentiment'] == 'netral']['processed_comments_for_wordcloud'].astype(str))

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("### Positif")
                if text_positif:
                    wc_positif = WordCloud(width=400, height=200, background_color='white', colormap='viridis', max_words=100).generate(text_positif)
                    fig_pos, ax_pos = plt.subplots(figsize=(4, 2))
                    ax_pos.imshow(wc_positif, interpolation='bilinear')
                    ax_pos.axis('off')
                    st.pyplot(fig_pos)
                else:
                    st.write("Tidak ada data sentimen positif.")

            with col2:
                st.write("### Negatif")
                if text_negatif:
                    wc_negatif = WordCloud(width=400, height=200, background_color='white', colormap='plasma', max_words=100).generate(text_negatif)
                    fig_neg, ax_neg = plt.subplots(figsize=(4, 2))
                    ax_neg.imshow(wc_negatif, interpolation='bilinear')
                    ax_neg.axis('off')
                    st.pyplot(fig_neg)
                else:
                    st.write("Tidak ada data sentimen negatif.")

            with col3:
                st.write("### Netral")
                if text_netral:
                    wc_netral = WordCloud(width=400, height=200, background_color='white', colormap='cividis', max_words=100).generate(text_netral)
                    fig_net, ax_net = plt.subplots(figsize=(4, 2))
                    ax_net.imshow(wc_netral, interpolation='bilinear')
                    ax_net.axis('off')
                    st.pyplot(fig_net)
                else:
                    st.write("Tidak ada data sentimen netral.")
        except FileNotFoundError:
            st.warning("File 'data_labelled.xlsx' tidak ditemukan. WordCloud mungkin tidak tersedia.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat atau memproses data untuk WordCloud: {e}")

    else:
        st.warning("Model, vektorizer, atau data split tidak dapat dimuat. Pastikan semua file .pkl dan kamus_normalisasi_lengkap.xlsx ada dan tidak rusak.")

elif menu_selection == "Prediksi Sentimen":
    st.header("Prediksi Sentimen")
    st.write("Masukkan komentar YouTube untuk memprediksi sentimennya (positif atau negatif).")
    user_input = st.text_area("Komentar Anda", "")
    if st.button("Prediksi"):
        if user_input:
            sentiment = predict_sentiment(user_input)
            st.write(f"Sentimen: **{sentiment}**")
        else:
            st.warning("Mohon masukkan komentar terlebih dahulu.")

st.sidebar.markdown("---")


