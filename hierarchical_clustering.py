# file: hierarchical_clustering_mall_customers_with_explanation.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Judul aplikasi
st.title("Hierarchical Clustering - Mall Customers")

# Upload file
st.subheader("1. Upload Dataset")
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file is not None:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)

    st.write("Data Awal:")
    st.dataframe(df)

    # Pilih fitur yang akan dipakai
    st.subheader("2. Pilih Fitur untuk Clustering")
    fitur_terpilih = st.multiselect(
        "Pilih kolom fitur:", 
        df.select_dtypes(include=[np.number]).columns.tolist(), 
        default=['Annual Income (k$)', 'Spending Score (1-100)']
    )

    if len(fitur_terpilih) >= 2:
        data = df[fitur_terpilih].values

        # Pilih metode linkage
        st.subheader("3. Pilih Metode Linkage")
        linkage_method = st.selectbox("Metode linkage:", ["single", "complete", "average", "ward"])

        # Tentukan jumlah cluster
        st.subheader("4. Tentukan Jumlah Cluster")
        n_clusters = st.slider("Jumlah cluster:", min_value=2, max_value=10, value=3)

        # Tombol untuk proses clustering
        if st.button("Proses Clustering"):
            # Buat linkage matrix
            linked = linkage(data, method=linkage_method)

            # Buat dendrogram
            st.subheader("5. Dendrogram")
            fig, ax = plt.subplots(figsize=(12, 6))
            dendrogram(linked,
                       orientation='top',
                       distance_sort='descending',
                       show_leaf_counts=True)
            st.pyplot(fig)

            # Penjelasan Dendrogram
            st.subheader("Penjelasan Dendrogram")
            st.write(
                "Dendrogram menunjukkan bagaimana data di-cluster berdasarkan jarak antar poin. "
                "Semakin dekat dua poin, semakin rendah posisi mereka dalam dendrogram. "
                "Cabang yang lebih dekat menunjukkan bahwa cluster tersebut lebih mirip satu sama lain."
            )

            # Menampilkan threshold untuk cluster
            st.write(f"Jumlah cluster yang dipilih: {n_clusters}")
            threshold = np.sort(linked[:, 2])[n_clusters - 2]  # Threshold jarak untuk membagi cluster
            st.write(f"Threshold untuk memotong dendrogram: {threshold:.2f}")

            # Garis pemotongan pada dendrogram
            st.write("Garis pemotongan untuk pembentukan cluster:")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            dendrogram(linked,
                       orientation='top',
                       distance_sort='descending',
                       show_leaf_counts=True,
                       color_threshold=threshold)
            ax2.axhline(y=threshold, c='r', linestyle='--')  # Garis pemotongan merah
            st.pyplot(fig2)

            # Membuat cluster
            clusters = fcluster(linked, n_clusters, criterion='maxclust')

            # Tambahkan cluster ke dataframe
            df['Cluster'] = clusters

            st.subheader("6. Data dengan Cluster:")
            st.dataframe(df)

            # Visualisasi hasil clustering
            st.subheader("7. Visualisasi Cluster")
            fig3, ax3 = plt.subplots()
            for cluster_id in np.unique(clusters):
                cluster_points = data[clusters == cluster_id]
                ax3.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")
            ax3.legend()
            ax3.set_xlabel(fitur_terpilih[0])
            ax3.set_ylabel(fitur_terpilih[1])
            ax3.set_title("Cluster Visualization")
            st.pyplot(fig3)

            # Penjelasan hasil clustering
            st.subheader("8. Penjelasan Setiap Cluster")
            cluster_summary = df.groupby('Cluster')[fitur_terpilih].agg(['count', 'mean']).reset_index()

            # Dapatkan rata-rata keseluruhan
            overall_means = df[fitur_terpilih].mean()

            for idx, row in cluster_summary.iterrows():
                cluster_id = int(row['Cluster'])
                count = int(row[(fitur_terpilih[0], 'count')])
                
                st.markdown(f"### Cluster {cluster_id}")
                st.write(f"Jumlah anggota: {count}")

                penjelasan = ""

                for fitur in fitur_terpilih:
                    cluster_mean = row[(fitur, 'mean')]
                    overall_mean = overall_means[fitur]

                    if cluster_mean > overall_mean * 1.1:
                        penjelasan += f"- Rata-rata {fitur} **tinggi** ({cluster_mean:.2f}) dibanding rata-rata ({overall_mean:.2f}).\n"
                    elif cluster_mean < overall_mean * 0.9:
                        penjelasan += f"- Rata-rata {fitur} **rendah** ({cluster_mean:.2f}) dibanding rata-rata ({overall_mean:.2f}).\n"
                    else:
                        penjelasan += f"- Rata-rata {fitur} **normal** ({cluster_mean:.2f}).\n"
                
                st.markdown(penjelasan)
    else:
        st.warning("Pilih minimal 2 fitur untuk clustering!")
else:
    st.info("Silakan upload file CSV untuk memulai.")
