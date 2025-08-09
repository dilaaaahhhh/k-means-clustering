import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt

st.title("Aplikasi K-Means Clustering + DBI")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Awal:", data.head())

    # Pilih kolom untuk clustering
    cols = st.multiselect("Pilih kolom fitur:", data.columns.tolist())
    if len(cols) >= 2:
        # Pilih jumlah cluster
        k = st.slider("Jumlah Cluster (k)", 2, 10, 3)

        # Proses K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        data['Cluster'] = kmeans.fit_predict(data[cols])

        # Hitung DBI
        dbi_score = davies_bouldin_score(data[cols], data['Cluster'])
        st.success(f"Nilai DBI: {dbi_score:.4f}")

        # Tampilkan hasil cluster
        st.write("Hasil Clustering:", data)

        # Visualisasi
        fig, ax = plt.subplots()
        scatter = ax.scatter(data[cols[0]], data[cols[1]], c=data['Cluster'], cmap='viridis', s=50)
        plt.xlabel(cols[0])
        plt.ylabel(cols[1])
        plt.title("Visualisasi Cluster")
        plt.colorbar(scatter)
        st.pyplot(fig)
