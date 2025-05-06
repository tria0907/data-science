import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.cluster import KMeans
from wordcloud import WordCloud

st.set_page_config(page_title="Facebook Review Analyzer", layout="wide")
st.title("📱 Facebook Review Analyzer & ML Explorer")

uploaded_file = st.file_uploader("📂 Upload Dataset Review Facebook (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset berhasil dimuat.")
    
    required_cols = ['content', 'score', 'thumbsUpCount', 'at', 'appVersion']
    if not all(col in df.columns for col in required_cols):
        st.error("❌ Dataset harus memiliki kolom: content, score, thumbsUpCount, at, appVersion")
    else:
        df = df[required_cols].dropna()
        df['review_length'] = df['content'].apply(lambda x: len(str(x).split()))
        df['at'] = pd.to_datetime(df['at'])
        df['month'] = df['at'].dt.to_period('M')

        # Filter interaktif (TIDAK menggunakan sidebar)
        st.subheader("🎛 Filter Data Review")
        min_score, max_score = int(df['score'].min()), int(df['score'].max())
        score_filter = st.slider("Filter berdasarkan skor", min_score, max_score, (min_score, max_score))
        date_filter = st.date_input("Rentang Tanggal", [df['at'].min(), df['at'].max()])

        df_filtered = df[(df['score'] >= score_filter[0]) & (df['score'] <= score_filter[1])]
        df_filtered = df_filtered[(df_filtered['at'] >= pd.to_datetime(date_filter[0])) & 
                                  (df_filtered['at'] <= pd.to_datetime(date_filter[1]))]

        st.subheader("📊 Preview Data Setelah Filter")
        st.dataframe(df_filtered.head())

        # Visualisasi
        st.subheader("📈 Distribusi Rating")
        fig, ax = plt.subplots()
        sns.countplot(x='score', data=df_filtered, ax=ax)
        st.pyplot(fig)

        st.subheader("📉 Panjang Ulasan terhadap Rating")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='score', y='review_length', data=df_filtered, ax=ax2)
        st.pyplot(fig2)

        st.subheader("📅 Tren Jumlah Ulasan per Bulan")
        monthly = df_filtered.groupby('month').size()
        fig3, ax3 = plt.subplots()
        monthly.plot(ax=ax3)
        ax3.set_ylabel("Jumlah Ulasan")
        st.pyplot(fig3)

        st.subheader("☁ Wordcloud Review Positif")
        text = ' '.join(df_filtered[df_filtered['score'] >= 4]['content'])
        wc = WordCloud(width=800, height=400).generate(text)
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.imshow(wc, interpolation='bilinear')
        ax4.axis('off')
        st.pyplot(fig4)

        st.divider()
        st.header("🧠 Machine Learning")

        # Klasifikasi
        st.subheader("🎯 Klasifikasi Review Positif/Negatif")
        df_clf = df_filtered[df_filtered['score'].isin([1, 2, 4, 5])].copy()
        df_clf['label'] = df_clf['score'].apply(lambda x: 1 if x >= 4 else 0)

        X = df_clf['content']
        y = df_clf['label']
        vectorizer = TfidfVectorizer(max_features=1000)
        X_vec = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, random_state=42)
        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred), language='text')

        # Clustering
        st.subheader("🧩 Clustering Review dengan KMeans")
        n_clusters = st.slider("Jumlah Cluster", 2, 5, 3, key="cluster_slider")
        X_clust = TfidfVectorizer(max_features=500).fit_transform(df_filtered['content'])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_filtered['cluster'] = kmeans.fit_predict(X_clust)

        fig5, ax5 = plt.subplots()
        sns.countplot(x='cluster', data=df_filtered, ax=ax5)
        st.pyplot(fig5)

        # Regresi
        st.subheader("📈 Prediksi Skor Berdasarkan Panjang Review & Likes")
        df_reg = df_filtered[['score', 'review_length', 'thumbsUpCount']].dropna()
        X_reg = df_reg[['review_length', 'thumbsUpCount']]
        y_reg = df_reg['score']

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, random_state=42)
        reg = LinearRegression().fit(X_train_r, y_train_r)
        pred_score = reg.predict(X_test_r)

        mse = mean_squared_error(y_test_r, pred_score)
        st.success(f"Mean Squared Error: {mse:.2f}")

        st.markdown("#### 🔮 Prediksi Skor dari Input")
        review_len = st.number_input("Panjang Review (kata)", min_value=0)
        thumbs = st.number_input("Jumlah Likes", min_value=0)
        if st.button("Prediksi Skor"):
            pred = reg.predict([[review_len, thumbs]])[0]
            st.info(f"Prediksi Skor Review: *{pred:.2f}*")