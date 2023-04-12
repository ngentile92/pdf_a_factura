
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from processing.extract_pdf_text import extract_pdf_text
import os


def process_text_cluster(text):
    concatenated_text = ' '.join(text.split('\n'))
    return concatenated_text.strip()

# Funciones extract_pdf_text y process_text ya proporcionadas

def vectorize_text(text_list):
    vectorizer = TfidfVectorizer()
    text_vectors = vectorizer.fit_transform(text_list)
    return text_vectors

def apply_kmeans_clustering(text_vectors, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(text_vectors)
    return kmeans.labels_

def clustering(pdf_paths, num_clusters):
    texts = [extract_pdf_text(pdf_path) for pdf_path in pdf_paths]
    preprocessed_texts = [process_text_cluster(text) for text in texts]
    text_vectors = vectorize_text(preprocessed_texts)
    cluster_labels = apply_kmeans_clustering(text_vectors, num_clusters)

    # Assign cluster labels to the corresponding PDF paths
    pdf_clusters = {pdf_path: label for pdf_path, label in zip(pdf_paths, cluster_labels)}
    
    return pdf_clusters

# function that receive the pdf_directory and the number of clusters and return a dictionary with the pdf name and the cluster number
def cluster_pdfs(pdf_directory, num_clusters):
    # Enumera todos los archivos en el directorio y filtra solo los archivos PDF
    pdf_filenames = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    # Crea una lista de rutas de archivos PDF
    pdf_paths = [os.path.join(pdf_directory, pdf_filename) for pdf_filename in pdf_filenames]

    pdf_clusters = clustering(pdf_paths, num_clusters)

    # delete everything before \\ in the path
    pdf_clusters = {k.split("\\")[-1]: v for k, v in pdf_clusters.items()}
    return pdf_clusters

if __name__ == "__main__":
    pdf_directory = "C:/Users/nagge/Desktop/exa_facturas"
    num_clusters = 4  # Ajustar este valor según tus necesidades
    pdf_clusters = cluster_pdfs(pdf_directory, num_clusters)
    print(pdf_clusters)