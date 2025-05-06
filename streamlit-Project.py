import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, silhouette_score

st.set_page_config(page_title="Smartphone ML Explorer", layout="wide")
st.title("📱 Smartphone Insight & Machine Learning Explorer")

uploaded_file = st.file_uploader("📂 Upload Dataset Smartphone (CSV)", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File berhasil dimuat.")
    st.dataframe(df.head())

    # Kolom numerik & label
    num_cols = df.select_dtypes(include='number').columns.tolist()

    st.header("📊 Eksplorasi Data")

    # Visualisasi: Boxplot Harga
    if 'Brand' in df.columns and 'Final Price' in df.columns:
        st.subheader("📦 Boxplot Harga per Brand")
        fig, ax = plt.subplots(figsize=(10,5))
        sns.boxplot(data=df, x='Brand', y='Final Price', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Korelasi fitur numerik
    st.subheader("📌 Korelasi Fitur Numerik")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.divider()

    st.header("🧠 Machine Learning")

    ### === 1. KLASIFIKASI Brand dari fitur numerik === ###
    st.subheader("🎯 Klasifikasi Brand Smartphone")

    # Pilih fitur numerik untuk klasifikasi
    clas_features = st.multiselect("Pilih fitur untuk klasifikasi", num_cols, default=['RAM', 'Storage'])

    if clas_features and 'Brand' in df.columns:
        df_clas = df[clas_features + ['Brand']].dropna()
        X = df_clas[clas_features]
        y = df_clas['Brand']
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, random_state=42)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = clf.score(X_test, y_test)
        st.success(f"Akurasi klasifikasi: {acc:.2f}")
        st.text("Label kelas:\n" + str(dict(zip(le.classes_, le.transform(le.classes_)))))

        # Prediksi interaktif
        st.markdown("#### 🔮 Prediksi Brand dari Input")
        input_vals = [st.number_input(f"{f}", value=0.0) for f in clas_features]
        pred = clf.predict([input_vals])[0]
        st.info(f"Prediksi Brand: **{le.inverse_transform([pred])[0]}**")

    st.divider()

    ### === 2. REGRESI Prediksi Harga === ###
    st.subheader("💰 Prediksi Harga Smartphone (Regresi)")

    if 'Final Price' in df.columns:
        reg_features = st.multiselect("Fitur untuk prediksi harga", num_cols[:-1], default=['RAM', 'Storage'])
        if reg_features:
            df_reg = df[reg_features + ['Final Price']].dropna()
            X = df_reg[reg_features]
            y = df_reg['Final Price']

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
            reg = RandomForestRegressor()
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            st.success(f"Mean Absolute Error: {mae:.2f}")

            # Prediksi interaktif
            st.markdown("#### 🔮 Prediksi Harga dari Input")
            reg_input = [st.number_input(f"{f}", value=0.0, key=f"reg_{f}") for f in reg_features]
            pred_price = reg.predict([reg_input])[0]
            st.info(f"Prediksi Harga: **Rp {pred_price:,.0f}**")

    st.divider()

    ### === 3. CLUSTERING === ###
    st.subheader("🧩 Clustering Smartphone")

    cluster_features = st.multiselect("Fitur numerik untuk clustering", num_cols, default=['RAM', 'Storage'])
    n_clusters = st.slider("Jumlah Cluster", 2, 5, 3)

    if cluster_features:
        df_cluster = df[cluster_features].dropna()
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_cluster)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data_scaled)

        st.write(f"Silhouette Score: **{silhouette_score(data_scaled, cluster_labels):.2f}**")
        df['Cluster'] = -1
        df.loc[df_cluster.index, 'Cluster'] = cluster_labels

        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=cluster_labels, palette='Set2', ax=ax3)
        plt.xlabel(cluster_features[0])
        plt.ylabel(cluster_features[1])
        st.pyplot(fig3)

        st.dataframe(df[['Brand'] + cluster_features + ['Cluster']].head(10))

st.markdown("""
<hr style="margin-top:50px;margin-bottom:20px">

<div style='background-color: #f0f0f0; padding: 15px; border-radius: 10px; text-align: center;'>
    <h4 style='color: #333;'>📱 <strong>Streamlit Smartphone Insight & Machine Learning Explorer</strong></h4>
    <p style='font-size:16px; color: #555;'>
        by <strong>Triani Yuli A</strong> &nbsp; | &nbsp; <strong>4D</strong> &nbsp; | &nbsp; <strong>Teknologi Informasi</strong>
    </p>
</div>
""", unsafe_allow_html=True)