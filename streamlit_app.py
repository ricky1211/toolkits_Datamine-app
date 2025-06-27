import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Import untuk PDF
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import base64
from datetime import datetime

# Konfigurasi halaman - HARUS DI AWAL!
st.set_page_config(
    page_title="DataMine UPB - Kelompok 3",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .main-header h3 {
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 Project Data Mining Kelompok 3 2025</h1>
    <h3>Teknik Informatika - TI.22.A4 - Universitas Pelita Bangsa</h3>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Menu Utama")
menu = st.sidebar.selectbox(
    "Pilih Fitur:",
    ["🏠 Dashboard", "📁 Upload Dataset", "🤖 Algoritma ML", "📈 Visualisasi", "📋 Laporan"]
)

# Fungsi untuk memuat data contoh
@st.cache_data
def load_sample_data():
    # Dataset Iris untuk klasifikasi
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # Dataset Boston Housing untuk regresi (simulasi karena deprecated)
    np.random.seed(42)
    n_samples = 500
    boston_df = pd.DataFrame({
        'CRIM': np.random.exponential(3, n_samples),
        'ZN': np.random.uniform(0, 100, n_samples),
        'INDUS': np.random.uniform(0, 30, n_samples),
        'CHAS': np.random.choice([0, 1], n_samples),
        'NOX': np.random.uniform(0.3, 0.9, n_samples),
        'RM': np.random.normal(6, 1, n_samples),
        'AGE': np.random.uniform(0, 100, n_samples),
        'DIS': np.random.exponential(3, n_samples),
        'RAD': np.random.choice(range(1, 25), n_samples),
        'TAX': np.random.uniform(150, 700, n_samples),
        'PTRATIO': np.random.uniform(12, 22, n_samples),
        'B': np.random.uniform(0, 400, n_samples),
        'LSTAT': np.random.uniform(1, 40, n_samples)
    })
    boston_df['MEDV'] = (boston_df['RM'] * 5 + np.random.normal(0, 5, n_samples)).clip(5, 50)
    
    return iris_df, boston_df

# Fungsi untuk validasi stratify
def can_stratify(y, test_size=0.2):
    """
    Cek apakah data dapat di-stratify dengan aman
    """
    try:
        unique_values = pd.Series(y).value_counts()
        min_samples = unique_values.min()
        total_samples = len(y)
        
        # Perlu minimal 2 sample per class untuk stratify
        # Dan minimal 1 sample per class di test set
        min_test_samples = max(1, int(total_samples * test_size))
        
        if min_samples < 2 or min_test_samples > min_samples:
            return False
        return True
    except:
        return False

# Fungsi untuk membuat PDF
def create_pdf_report(nama, npm, kelas, tanggal, df, algoritma_used, kesimpulan):
    """
    Membuat laporan dalam format PDF
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    # Konten PDF
    story = []
    
    # Header
    story.append(Paragraph("LAPORAN PRAKTIKUM DATA MINING 2025", title_style))
    story.append(Paragraph("Teknik Informatika - Universitas Pelita Bangsa", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Garis pemisah
    story.append(Paragraph("_" * 80, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Identitas Mahasiswa
    story.append(Paragraph("IDENTITAS MAHASISWA", heading_style))
    identity_data = [
        ["Nama:", nama],
        ["NPM:", npm],
        ["Kelas:", kelas],
        ["Tanggal:", str(tanggal)]
    ]
    identity_table = Table(identity_data, colWidths=[1.5*inch, 4*inch])
    identity_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(identity_table)
    story.append(Spacer(1, 20))
    
    # Informasi Dataset
    story.append(Paragraph("INFORMASI DATASET", heading_style))
    dataset_info = [
        ["Jumlah Baris:", str(df.shape[0])],
        ["Jumlah Kolom:", str(df.shape[1])],
        ["Missing Values:", str(df.isnull().sum().sum())],
        ["Kolom Dataset:", ", ".join(df.columns.tolist()[:5]) + ("..." if len(df.columns) > 5 else "")]
    ]
    dataset_table = Table(dataset_info, colWidths=[2*inch, 4*inch])
    dataset_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(dataset_table)
    story.append(Spacer(1, 20))
    
    # Statistik Deskriptif (hanya untuk kolom numerik)
    story.append(Paragraph("STATISTIK DESKRIPTIF", heading_style))
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Ambil 5 kolom pertama
    if not numeric_cols.empty:
        desc_stats = df[numeric_cols].describe()
        
        # Konversi ke format yang bisa ditampilkan di tabel
        stats_data = [["Statistik"] + list(desc_stats.columns)]
        for index in desc_stats.index:
            row = [index] + [f"{val:.2f}" if isinstance(val, (int, float)) else str(val) 
                           for val in desc_stats.loc[index]]
            stats_data.append(row)
        
        stats_table = Table(stats_data)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(stats_table)
    else:
        story.append(Paragraph("Tidak ada kolom numerik untuk statistik deskriptif", normal_style))
    
    story.append(Spacer(1, 20))
    
    # Algoritma yang digunakan
    story.append(Paragraph("ALGORITMA YANG DIGUNAKAN", heading_style))
    if algoritma_used:
        for algo in algoritma_used:
            story.append(Paragraph(f"• {algo}", normal_style))
    else:
        story.append(Paragraph("Belum ada algoritma yang dipilih", normal_style))
    story.append(Spacer(1, 20))
    
    # Kesimpulan
    story.append(Paragraph("KESIMPULAN", heading_style))
    if kesimpulan:
        story.append(Paragraph(kesimpulan, normal_style))
    else:
        story.append(Paragraph("Belum ada kesimpulan yang ditulis", normal_style))
    story.append(Spacer(1, 30))
    
    # Footer
    story.append(Paragraph("_" * 80, styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Laporan ini dibuat menggunakan aplikasi Data Mining Praktikum 2025", 
                          styles['Normal']))
    story.append(Paragraph("Universitas Pelita Bangsa - Teknik Informatika", styles['Normal']))
    story.append(Paragraph(f"Dibuat pada: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", 
                          styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# Dashboard
if menu == "🏠 Dashboard":
    st.header("Dashboard Utama")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Algoritma Tersedia</h4>
            <h2>7</h2>
            <p>Regresi, Klasifikasi, Clustering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📁 Format Data</h4>
            <h2>CSV</h2>
            <p>Upload dan analisis data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📈 Visualisasi</h4>
            <h2>Interaktif</h2>
            <p>Grafik dan plot dinamis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>📋 Export</h4>
            <h2>PDF</h2>
            <p>Laporan dan poster</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Informasi Algoritma
    st.subheader("🧠 Algoritma yang Tersedia")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Regresi:**
        - Linear Regression
        - Logistic Regression
        
        **Klasifikasi:**
        - Naive Bayes
        - Support Vector Machine (SVM)
        """)
    
    with col2:
        st.markdown("""
        **Klasifikasi Lanjutan:**
        - K-Nearest Neighbors (KNN)
        - Decision Tree
        
        **Clustering:**
        - K-Means Clustering
        """)

# Upload Dataset
elif menu == "📁 Upload Dataset":
    st.header("Upload Dataset")
    
    # Pilihan dataset
    data_option = st.radio(
        "Pilih sumber data:",
        ["Upload file CSV", "Gunakan dataset contoh"]
    )
    
    df = None
    
    if data_option == "Upload file CSV":
        uploaded_file = st.file_uploader(
            "Pilih file CSV", 
            type=['csv'],
            help="Upload file CSV untuk dianalisis"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Dataset berhasil dimuat! Shape: {df.shape}")
                
                # Simpan ke session state
                st.session_state['df'] = df
                
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
    
    else:
        dataset_choice = st.selectbox(
            "Pilih dataset contoh:",
            ["Iris (Klasifikasi)", "Boston Housing (Regresi)"]
        )
        
        iris_df, boston_df = load_sample_data()
        
        if dataset_choice == "Iris (Klasifikasi)":
            df = iris_df
            st.info("📊 Dataset Iris dimuat untuk klasifikasi bunga")
        else:
            df = boston_df
            st.info("🏠 Dataset Boston Housing dimuat untuk prediksi harga")
        
        st.session_state['df'] = df
    
    # Tampilkan preview data
    if df is not None:
        st.subheader("👀 Preview Data")
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Info Dataset")
            st.write(f"**Jumlah Baris:** {df.shape[0]}")
            st.write(f"**Jumlah Kolom:** {df.shape[1]}")
            st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        
        with col2:
            st.subheader("📈 Statistik Deskriptif")
            st.dataframe(df.describe())

# Algoritma ML
elif menu == "🤖 Algoritma ML":
    st.header("Implementasi Algoritma Machine Learning")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu di menu 'Upload Dataset'")
    else:
        df = st.session_state['df']
        
        # Pilih algoritma
        algorithm = st.selectbox(
            "Pilih Algoritma:",
            ["Linear Regression", "Logistic Regression", "Naive Bayes", 
             "SVM", "KNN", "Decision Tree", "K-Means"]
        )
        
        # Pilih kolom untuk analisis
        st.subheader("⚙️ Konfigurasi Model")
        
        if algorithm in ["Linear Regression"]:
            # Regresi
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("❌ Dataset memerlukan minimal 2 kolom numerik untuk regresi")
            else:
                target_col = st.selectbox("Pilih kolom target (y):", numeric_cols)
                feature_cols = st.multiselect(
                    "Pilih kolom fitur (X):", 
                    [col for col in numeric_cols if col != target_col]
                )
                
                if len(feature_cols) > 0 and st.button("🚀 Jalankan Model"):
                    try:
                        X = df[feature_cols].dropna()
                        y = df[target_col].loc[X.index]
                        
                        if len(X) < 5:
                            st.error("❌ Data terlalu sedikit untuk analisis (minimal 5 baris)")
                        else:
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.2, random_state=42
                            )
                            
                            # Model
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            
                            # Prediksi
                            y_pred = model.predict(X_test)
                            
                            # Evaluasi
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            st.success("✅ Model berhasil dijalankan!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Mean Squared Error", f"{mse:.4f}")
                            with col2:
                                st.metric("R² Score", f"{r2:.4f}")
                            
                            # Plot hasil
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(y_test, y_pred, alpha=0.6)
                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                            ax.set_xlabel('Actual Values')
                            ax.set_ylabel('Predicted Values')
                            ax.set_title('Actual vs Predicted Values')
                            st.pyplot(fig)
                    
                    except Exception as e:
                        st.error(f"❌ Error menjalankan model: {str(e)}")
        
        elif algorithm in ["Logistic Regression", "Naive Bayes", "SVM", "KNN", "Decision Tree"]:
            # Klasifikasi
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_cols = categorical_cols + numeric_cols
            
            if len(all_cols) < 2:
                st.error("❌ Dataset memerlukan minimal 2 kolom untuk klasifikasi")
            else:
                target_col = st.selectbox("Pilih kolom target (y):", all_cols)
                feature_cols = st.multiselect(
                    "Pilih kolom fitur (X):", 
                    [col for col in all_cols if col != target_col]
                )
                
                if len(feature_cols) > 0 and st.button("🚀 Jalankan Model"):
                    try:
                        # Preprocessing
                        X = df[feature_cols].copy()
                        y = df[target_col].copy()
                        
                        # Remove missing values
                        mask = ~(X.isnull().any(axis=1) | y.isnull())
                        X = X[mask]
                        y = y[mask]
                        
                        if len(X) < 5:
                            st.error("❌ Data terlalu sedikit untuk analisis (minimal 5 baris)")
                        else:
                            # Handle categorical variables
                            le_dict = {}
                            for col in X.columns:
                                if X[col].dtype == 'object':
                                    le = LabelEncoder()
                                    X[col] = le.fit_transform(X[col].astype(str))
                                    le_dict[col] = le
                            
                            if y.dtype == 'object':
                                le_target = LabelEncoder()
                                y = le_target.fit_transform(y.astype(str))
                            
                            # Check if we can stratify
                            use_stratify = can_stratify(y, test_size=0.2)
                            
                            # Split data
                            if use_stratify:
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, random_state=42, stratify=y
                                )
                            else:
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, random_state=42
                                )
                                st.info("ℹ️ Stratifikasi tidak dapat dilakukan karena distribusi kelas tidak seimbang")
                            
                            # Scaling untuk SVM dan KNN
                            if algorithm in ["SVM", "KNN"]:
                                scaler = StandardScaler()
                                X_train = scaler.fit_transform(X_train)
                                X_test = scaler.transform(X_test)
                            
                            # Model selection
                            if algorithm == "Logistic Regression":
                                model = LogisticRegression(random_state=42, max_iter=1000)
                            elif algorithm == "Naive Bayes":
                                model = GaussianNB()
                            elif algorithm == "SVM":
                                model = SVC(random_state=42)
                            elif algorithm == "KNN":
                                k = st.slider("Pilih nilai K:", 1, min(20, len(X_train)), 5)
                                model = KNeighborsClassifier(n_neighbors=k)
                            elif algorithm == "Decision Tree":
                                model = DecisionTreeClassifier(random_state=42)
                            
                            # Training
                            model.fit(X_train, y_train)
                            
                            # Prediksi
                            y_pred = model.predict(X_test)
                            
                            # Evaluasi
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            st.success("✅ Model berhasil dijalankan!")
                            st.metric("Accuracy", f"{accuracy:.4f}")
                            
                            # Classification Report
                            st.subheader("📊 Classification Report")
                            try:
                                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df)
                            except Exception as e:
                                st.warning(f"⚠️ Tidak dapat menampilkan classification report: {str(e)}")
                            
                            # Confusion Matrix
                            st.subheader("🔍 Confusion Matrix")
                            try:
                                cm = confusion_matrix(y_test, y_pred)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                ax.set_title('Confusion Matrix')
                                ax.set_xlabel('Predicted')
                                ax.set_ylabel('Actual')
                                st.pyplot(fig)
                            except Exception as e:
                                st.warning(f"⚠️ Tidak dapat menampilkan confusion matrix: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"❌ Error menjalankan model: {str(e)}")
        
        elif algorithm == "K-Means":
            # Clustering
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("❌ Dataset memerlukan minimal 2 kolom numerik untuk clustering")
            else:
                feature_cols = st.multiselect(
                    "Pilih kolom untuk clustering:", 
                    numeric_cols
                )
                
                if len(feature_cols) >= 2:
                    k = st.slider("Pilih jumlah cluster (K):", 2, min(10, len(df)//2), 3)
                    
                    if st.button("🚀 Jalankan Clustering"):
                        try:
                            X = df[feature_cols].dropna()
                            
                            if len(X) < k:
                                st.error(f"❌ Data terlalu sedikit untuk {k} cluster (ada {len(X)} baris)")
                            else:
                                # Scaling
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)
                                
                                # K-Means
                                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                clusters = kmeans.fit_predict(X_scaled)
                                
                                # Tambahkan hasil clustering ke dataframe
                                df_result = X.copy()
                                df_result['Cluster'] = clusters
                                
                                st.success("✅ Clustering berhasil!")
                                
                                # Visualisasi (untuk 2 fitur pertama)
                                if len(feature_cols) >= 2:
                                    fig = px.scatter(
                                        df_result, 
                                        x=feature_cols[0], 
                                        y=feature_cols[1],
                                        color='Cluster',
                                        title=f'K-Means Clustering (K={k})'
                                    )
                                    st.plotly_chart(fig)
                                
                                # Tampilkan hasil
                                st.subheader("📊 Hasil Clustering")
                                st.dataframe(df_result.head(10))
                                
                                # Cluster distribution
                                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                                st.subheader("📈 Distribusi Cluster")
                                st.bar_chart(cluster_counts)
                        
                        except Exception as e:
                            st.error(f"❌ Error menjalankan clustering: {str(e)}")

# Visualisasi
elif menu == "📈 Visualisasi":
    st.header("Visualisasi Data")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu di menu 'Upload Dataset'")
    else:
        df = st.session_state['df']
        
        # Pilih jenis visualisasi
        viz_type = st.selectbox(
            "Pilih jenis visualisasi:",
            ["Histogram", "Scatter Plot", "Box Plot", "Correlation Heatmap", "Bar Chart"]
        )
        
        try:
            if viz_type == "Histogram":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    col = st.selectbox("Pilih kolom:", numeric_cols)
                    
                    fig = px.histogram(df, x=col, title=f'Histogram of {col}')
                    st.plotly_chart(fig)
                else:
                    st.warning("⚠️ Tidak ada kolom numerik untuk histogram")
            
            elif viz_type == "Scatter Plot":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("Pilih kolom X:", numeric_cols)
                    with col2:
                        y_col = st.selectbox("Pilih kolom Y:", [col for col in numeric_cols if col != x_col])
                    
                    color_col = None
                    if len(df.columns) > 2:
                        color_col = st.selectbox(
                            "Pilih kolom untuk warna (opsional):", 
                            ["None"] + [col for col in df.columns if col not in [x_col, y_col]]
                        )
                        if color_col == "None":
                            color_col = None
                    
                    fig = px.scatter(
                        df, 
                        x=x_col, 
                        y=y_col, 
                        color=color_col,
                        title=f'Scatter Plot: {x_col} vs {y_col}'
                    )
                    st.plotly_chart(fig)
                else:
                    st.warning("⚠️ Perlu minimal 2 kolom numerik untuk scatter plot")
            
            elif viz_type == "Box Plot":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    col = st.selectbox("Pilih kolom:", numeric_cols)
                    
                    fig = px.box(df, y=col, title=f'Box Plot of {col}')
                    st.plotly_chart(fig)
                else:
                    st.warning("⚠️ Tidak ada kolom numerik untuk box plot")
            
            elif viz_type == "Correlation Heatmap":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    corr = df[numeric_cols].corr()
                    
                    fig = px.imshow(
                        corr,
                        labels=dict(x="Kolom", y="Kolom", color="Korelasi"),
                        x=numeric_cols,
                        y=numeric_cols,
                        title="Correlation Heatmap"
                    )
                    st.plotly_chart(fig)
                else:
                    st.warning("⚠️ Perlu minimal 2 kolom numerik untuk heatmap")
            
            elif viz_type == "Bar Chart":
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                if categorical_cols:
                    col = st.selectbox("Pilih kolom kategorikal:", categorical_cols)
                    
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = ['Category', 'Count']
                    
                    fig = px.bar(
                        value_counts, 
                        x='Category', 
                        y='Count',
                        title=f'Distribution of {col}'
                    )
                    st.plotly_chart(fig)
                else:
                    st.warning("⚠️ Tidak ada kolom kategorikal untuk bar chart")
        
        except Exception as e:
            st.error(f"❌ Error membuat visualisasi: {str(e)}")

# Laporan
elif menu == "📋 Laporan":
    st.header("📋 Buat Laporan Analisis")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu di menu 'Upload Dataset'")
    else:
        df = st.session_state['df']
        
        st.subheader("Identitas Mahasiswa")
        col1, col2 = st.columns(2)
        
        with col1:
            nama = st.text_input("Nama Lengkap", "")
            nim = st.text_input("nim", "")
        
        with col2:
            kelas = st.text_input("Kelas", "")
            tanggal = st.date_input("")
        
        st.subheader("Algoritma yang Digunakan")
        algoritma_used = st.multiselect(
            "Pilih algoritma yang digunakan:",
            ["Linear Regression", "Logistic Regression", "Naive Bayes", 
             "SVM", "KNN", "Decision Tree", "K-Means"]
        )
        
        st.subheader("Kesimpulan")
        kesimpulan = st.text_area(
            "Tulis kesimpulan analisis Anda:",
            "Berdasarkan analisis yang telah dilakukan, dapat disimpulkan bahwa..."
        )
        
        if st.button("📥 Generate PDF Report"):
            with st.spinner("Membuat laporan PDF..."):
                try:
                    pdf_buffer = create_pdf_report(
                        nama, nim, kelas, tanggal, df, algoritma_used, kesimpulan
                    )
                    
                    st.success("✅ Laporan berhasil dibuat!")
                    
                    # Download button
                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_buffer,
                        file_name=f"Laporan_Data_Mining_{nama}_{tanggal}.pdf",
                        mime="application/pdf"
                    )
                
                except Exception as e:
                    st.error(f"❌ Error membuat laporan PDF: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🔍 Project Data Mining 2025 - Kelompok 3 - TI.22.A4 - Universitas Pelita Bangsa</p>
    <p>Dibuat dengan Streamlit | © 2025</p>
</div>
""", unsafe_allow_html=True)
