import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="K-Means Clustering DBI", layout="wide")

st.title("📊 K-Means Clustering dengan DBI & Visualisasi")

# Upload data CSV
uploaded_file = st.file_uploader("📂 Upload file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Awal")
    st.dataframe(df)

    # Pilih kolom untuk clustering
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    features = st.multiselect("Pilih Kolom untuk Clustering", numeric_columns, default=numeric_columns[:2])

    if len(features) < 2:
        st.warning("Pilih minimal 2 kolom untuk clustering")
    else:
        # Tentukan jumlah cluster
        k = st.slider("Jumlah Cluster (K)", 2, 10, 3)

        # Normalisasi data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[features])

        # Jalankan K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)

        # Hitung DBI
        dbi_score = davies_bouldin_score(scaled_data, df['Cluster'])
        st.metric("Nilai Davies-Bouldin Index (DBI)", f"{dbi_score:.4f}")

        # Tampilkan data dengan cluster
        st.subheader("📊 Data dengan Label Cluster")
        st.dataframe(df)

        # Visualisasi cluster
        st.subheader("📈 Visualisasi Cluster")
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            df[features[0]], df[features[1]],
            c=df['Cluster'], cmap='viridis', s=50
        )
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title("Visualisasi Cluster")
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)

        # Simpan hasil
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Download Hasil CSV", csv, "hasil_clustering.csv", "text/csv")
else:
    st.info("Silakan upload file CSV untuk mulai.")
