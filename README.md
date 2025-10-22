# ⚔️ Sun Tzu RAG Chatbot

Sun Tzu'nun Savaş Sanatı eserine dayalı RAG (Retrieval-Augmented Generation) chatbot projesi. Akbank GenAI Bootcamp kapsamında geliştirilmiştir.

## 🎯 Proje Amacı

Bu proje, Sun Tzu'nun Savaş Sanatı eserinin Türkçe çevirisini kullanarak, kullanıcıların savaş stratejisi, liderlik ve taktik konularında sorular sorabileceği akıllı bir chatbot geliştirmeyi amaçlamaktadır.

## 🚀 Hızlı Başlangıç

### 1. Repository'yi Klonlayın
```bash
git clone https://github.com/gorkemgok0/rag-chatbot-artofwar-suntzu.git
cd rag-chatbot-artofwar-suntzu
```

### 2. Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. API Key'inizi Ayarlayın
`.env` dosyası oluşturun:
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### 4. Uygulamayı Başlatın
```bash
streamlit run web/app.py
```

Uygulama `http://localhost:8501` adresinde açılacaktır.

## 🛠️ Teknoloji Stack

- **🤖 LLM**: Google Gemini 1.5 Flash
- **🧠 Embedding**: Sentence Transformers (multilingual-MiniLM-L12-v2)
- **🗄️ Vector Database**: ChromaDB
- **🚀 Web Framework**: Streamlit
- **📚 Veri**: Sun Tzu'nun Savaş Sanatı (Türkçe)

## 📁 Proje Yapısı

```
rag-chatbot/
├── 📁 data/                    # Veri dosyaları (gitignore'da)
├── 📁 src/                     # Ana kod dosyaları
│   ├── embed_store.py         # Embedding ve ChromaDB scripti
│   └── rag_pipeline.py        # RAG pipeline test scripti
├── 📁 web/                     # Web arayüzü
│   └── app.py                 # Streamlit uygulaması
├── 📁 notebooks/               # Jupyter notebook'lar
│   ├── data_prep.py           # Veri hazırlama scripti
│   └── data_prep.ipynb        # Detaylı süreç notebook'u
├── 📄 requirements.txt        # Python bağımlılıkları
├── 📄 .env                    # API key (gitignore'da)
├── 📄 .gitignore              # Git ignore kuralları
└── 📄 README.md               # Bu dosya
```

## 🔧 Detaylı Kurulum

### Gereksinimler
- Python 3.8+
- Google API Key (Gemini için)

### Adım Adım Kurulum

1. **Repository'yi klonlayın**
2. **Virtual environment oluşturun** (isteğe bağlı)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # veya
   venv\Scripts\activate     # Windows
   ```

3. **Bağımlılıkları yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

4. **API Key'inizi ayarlayın**
   - [Google AI Studio](https://aistudio.google.com/) hesabı oluşturun
   - API key alın
   - `.env` dosyasına ekleyin:
     ```
     GOOGLE_API_KEY=your_actual_api_key_here
     ```

## 📊 Veri Seti

Bu proje, Hugging Face `umithy/sun_tzu_savas_sanati` veri setini kullanır:
- **Kaynak**: Sun Tzu'nun Savaş Sanatı (Türkçe çeviri)
- **İçerik**: 48 metin parçası, savaş stratejisi ve liderlik öğütleri
- **Format**: Paragraf bazlı, RAG için optimize edilmiş

## 🧠 RAG Mimarisi

### Retriever (Arama)
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Database**: ChromaDB (kalıcı veritabanı)
- **Embedding Boyutu**: 384 boyut

### Generator (Üretim)
- **LLM**: Google Gemini 1.5 Flash
- **Prompt Engineering**: Bağlama dayalı cevap üretimi
- **Dil**: Türkçe

## 🎮 Kullanım

### Web Arayüzü
```bash
streamlit run web/app.py
```

### Komut Satırı Testi
```bash
python src/rag_pipeline.py
```

### Örnek Sorular
- "Savaşta strateji nasıl belirlenir?"
- "Düşman nasıl yenilir?"
- "Ordunun moralini nasıl yükseltirsin?"
- "Kaynakları nasıl yönetirsin?"
- "Liderlik nasıl olmalıdır?"

## 🔒 Güvenlik

- ✅ API key'ler `.env` dosyasında saklanır
- ✅ `.env` dosyası `.gitignore`'da
- ✅ Veri dosyaları GitHub'a yüklenmez
- ✅ ChromaDB veritabanı lokal olarak saklanır

## 📈 Özellikler

- **🎯 Doğru Cevap**: Sun Tzu'nun öğretilerine dayalı yanıtlar
- **⚡ Hızlı**: Cache'lenmiş embedding'ler
- **🔄 Gerçek Zamanlı**: Anında yanıt alma
- **📱 Modern UI**: Streamlit ile responsive tasarım
- **🔍 Şeffaf**: Kullanılan kaynak metinleri görüntüleme

## 🚀 Deployment

### Streamlit Cloud (Önerilen)
1. GitHub repository'nizi Streamlit Cloud'a bağlayın
2. Environment variables'da `GOOGLE_API_KEY` ekleyin
3. Deploy edin

### Hugging Face Spaces
1. Hugging Face Spaces'te yeni Space oluşturun
2. Repository'nizi bağlayın
3. Environment variables ayarlayın

### Local Development
```bash
# Geliştirme ortamı
streamlit run web/app.py --server.port 8501
```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👨‍💻 Geliştirici

**Görkem Gök** - [GitHub](https://github.com/gorkemgok0)

## 🙏 Teşekkürler

- **Akbank GenAI Bootcamp** - Proje kapsamı
- **Hugging Face** - Veri seti ve modeller
- **Google** - Gemini API
- **Streamlit** - Web framework