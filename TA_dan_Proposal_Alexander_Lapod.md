# LAPORAN TUGAS AKHIR DAN PROPOSAL

## BAGIAN 1: LAPORAN TUGAS AKHIR
*(NEW) TA_160423232.docx*

### APLIKASI DETEKSI RISIKO MELANOMA MENGGUNAKAN WEIGHTED ENSEMBLE CNN DENGAN OPTIMASI ALGORITMA GENETIK

**Disusun Oleh:**
Alexander Fabiano Joynard Lapod  
NRP : 160423232  

**PROGRAM STUDI TEKNIK INFORMATIKA**  
**DATA SCIENCE AND ARTIFICIAL INTELLIGENCE**  
**FAKULTAS TEKNIK UNIVERSITAS SURABAYA**  
**Mei, 2026**

---

### ABSTRAK
Melanoma adalah jenis kanker kulit yang paling berbahaya dengan tingkat fatalitas tinggi, yang diagnosis dininya sering terhambat oleh kemiripan visual dengan lesi jinak dan keterbatasan akses fasilitas kesehatan. Pengembangan sistem deteksi berbasis kecerdasan buatan, khususnya melalui aplikasi hybrid mobile, menjadi solusi untuk skrining awal yang praktis. Tantangan utama meliputi performa klasifikasi pada dataset yang sangat tidak seimbang (HAM10000) dan perlunya efisiensi komputasi untuk perangkat mobile. 

Penelitian ini bertujuan membangun sistem deteksi tingkat risiko melanoma yang andal dan mudah diakses, menggunakan pendekatan Weighted Ensemble CNN yang menggabungkan tiga arsitektur pretrained (ResNet50, DenseNet121, dan EfficientNetB0). Untuk mengatasi kelemahan simple averaging dan memaksimalkan performa pada data yang tidak seimbang, optimasi bobot model ensemble dilakukan secara heuristik menggunakan Algoritma Genetik dengan metrik evaluasi utama F1-Score. Proses preprocessing mencakup segmentasi (Otsu's Thresholding) dan restorasi citra (Blackhat dan Inpainting Telea) untuk mengatasi noise seperti rambut. Sistem ini diimplementasikan dalam arsitektur client-server untuk memastikan kelancaran pada perangkat hybrid mobile. Hasil prediksi disajikan dalam bentuk tiga tingkatan risiko (Rendah, Sedang, Tinggi), menjadikannya lebih komunikatif bagi pengguna sebagai alat deteksi dini.

**Kata kunci:** Melanoma, Deteksi Risiko, Weighted Ensemble CNN, Optimasi Algoritma Genetik, Hybrid Mobile, F-1 Score

---

### KATA PENGANTAR
Puji syukur penulis panjatkan ke hadirat Tuhan Yang Maha Esa karena berkat dan rahmat-Nya, penulis dapat menyelesaikan penulisan Tugas Akhir yang berjudul “Aplikasi Deteksi Risiko Melanoma Menggunakan Weighted Ensemble CNN dengan Optimasi Algoritma Genetik”.

Tugas akhir ini dapat disusun dengan baik berkat dukungan dari berbagai pihak. Oleh karena itu, penulis mengucapkan terima kasih kepada:
1. Ibu Dr. Monica Widiasri., selaku dosen pembimbing 1 yang telah membimbing, membantu, dan memberikan dukungan kepada penulis dalam pembuatan tugas akhir ini.
2. Bapak Prof. Dr. Joko Siswantoro., selaku dosen pembimbing 2 yang telah membimbing, membantu, dan memberikan dukungan kepada penulis dalam pembuatan tugas akhir ini.
3. Orang tua yang selalu memberikan semangat dan dukungan penuh kepada penulis dalam menyelesaikan tugas akhir ini.
4. Teman-teman yang selalu memberikan dukungan dan semangat kepada penulis.

Penulis menyadari bahwa masih terdapat banyak kekurangan dalam penulisan tugas akhir ini. Oleh karena itu, penulis mengharapkan kritik dan saran yang membangun sebagai bahan evaluasi untuk kedepannya. Semoga tugas akhir ini dapat meningkatkan wawasan pembaca.

Surabaya, 18 May 2026  
Penulis  

---

### DAFTAR ISI
- ABSTRAK 
- ABSTRACT 
- KATA PENGANTAR 
- DAFTAR ISI 
- DAFTAR GAMBAR 
- DAFTAR TABEL 
- DAFTAR LISTING 
- BAB 1 PENDAHULUAN
  - 1.1 Latar Belakang
  - 1.2 Rumusan Masalah
  - 1.3 Tujuan
  - 1.4 Manfaat
  - 1.5 Ruang Lingkup
  - 1.6 Metodologi Penelitian
  - 1.7 Sistematika Penulisan
- BAB 2 LANDASAN TEORI
  - 2.1 Kanker Kulit Melanoma
  - 2.2 Convolutional Neural Network (CNN)
    - 2.2.1 Transfer Learning
    - 2.2.2 Arsitektur Pretrained CNN
  - 2.3 Weighted Ensemble CNN
  - 2.4 Optimasi Algoritma Genetik
  - 2.5 Dataset dan Preprocessing Citra
    - 2.5.1 Dataset HAM10000
  - 2.6 Penentuan Tingkat Risiko
  - 2.7 Evaluasi
  - 2.8 Framework dan Library
- BAB 3 ANALISIS SISTEM
  - 3.1 Kondisi Saat Ini
  - 3.2 Analisis Sistem Sejenis
  - 3.3 Analisis Permasalahan
  - 3.4 Analisis Kebutuhan Sistem
- DAFTAR PUSTAKA
- LAMPIRAN A

---

### BAB 1 PENDAHULUAN
Bab 1 menjelaskan secara umum mengenai latar belakang permasalahan, rumusan masalah, tujuan, manfaat, serta ruang lingkup dari tugas akhir ini. Bab ini juga membahas mengenai metodologi penelitian yang dilakukan serta sistematika penulisan laporan tugas akhir mengenai sistem deteksi melanoma.

#### 1.1 Latar Belakang
Kanker kulit adalah salah satu penyakit dengan jumlah kasus yang terus meningkat di seluruh dunia dan termasuk dalam kelompok kanker yang paling sering didiagnosis. Berdasarkan data World Health Organization (WHO), terdapat lebih dari 1,5 juta kasus baru kanker kulit pada tahun 2022. Dari berbagai jenis kanker kulit tersebut, melanoma merupakan jenis yang paling berbahaya karena memiliki tingkat fatalitas tinggi, meskipun hanya mencakup sebagian kecil dari total kasus. Hal ini disebabkan oleh kemampuan sel melanoma untuk menyebar dengan cepat ke organ tubuh lain. Secara tradisional, diagnosis melanoma dilakukan oleh dokter spesialis kulit melalui observasi visual dan dermoskopi. Namun, proses ini memiliki berbagai tantangan, antara lain kemiripan karakteristik visual antara lesi melanoma dan lesi jinak seperti nevus, serta adanya potensi bias subjektivitas pengamat (Abbas et al., 2011).

Karena permasalahan tersebut, pengembangan sistem deteksi berbasis kecerdasan buatan telah banyak dilakukan. Pendekatan ini dapat membantu proses deteksi dan klasifikasi kanker kulit dengan lebih efisien. Sistem berbasis kecerdasan buatan dapat mengurangi beban kerja tenaga medis dan mempercepat proses analisis dengan tingkat akurasi yang cukup tinggi (Kadeem & Mosa, 2024; Salian & Sawarkar, 2022). Metode yang digunakan juga dapat mengenali pola visual pada citra kulit secara otomatis, sehingga dapat membantu dalam proses pengambilan keputusan dengan lebih cepat dan adil.

Di Indonesia, kondisi ini menjadi semakin serius karena keterbatasan akses terhadap fasilitas kesehatan, kurangnya tenaga kesehatan, jarak tempuh menuju layanan medis yang jauh, dan rendahnya kesadaran masyarakat menyebabkan deteksi dini sering terlambat dilakukan. Menurut data, rasio dokter di Indonesia masih relatif rendah, yaitu sekitar 0,47 dokter per 1.000 penduduk, dan distribusinya tidak merata antar wilayah (Kementerian Kesehatan RI, 2023). Selain itu, ketersediaan fasilitas pelayanan kesehatan dasar juga masih belum sepenuhnya memenuhi kebutuhan masyarakat di berbagai daerah (Kementerian Kesehatan RI, 2022).

Seiring perkembangan teknologi, implementasi sistem deteksi juga mulai diarahkan ke perangkat mobile agar lebih mudah diakses masyarakat. Penggunaan aplikasi berbasis smartphone dapat menjadi solusi deteksi dini yang praktis, terutama bagi masyarakat yang memiliki keterbatasan akses ke tenaga medis. Namun demikian, penggunaan kamera ponsel juga menghadirkan tantangan baru, seperti variasi kualitas citra akibat pencahayaan yang tidak stabil, serta adanya gangguan seperti rambut dan noda pada kulit. Diperlukan proses preprocessing citra yang tepat agar kualitas input tetap mendukung kinerja model. Selain tantangan tersebut, implementasi pada smartphone juga dibatasi oleh spesifikasi perangkat yang bervariasi. Pendekatan model yang kompleks memang dapat memberikan performa tinggi, namun membutuhkan komputasi dan memori yang besar jika ditargetkan untuk perangkat mobile (Kousis et al., 2022). Jika berjalan secara lokal, hal ini berisiko dapat menyebabkan sistem menjadi lambat, cepat menguras baterai, bahkan mengalami kegagalan sistem pada perangkat dengan spesifikasi rendah. Untuk mengatasi hambatan tersebut, dibutuhkan penggunaan arsitektur client-server untuk menjaga efisiensi dan kenyamanan pengguna.

Berbagai penelitian terdahulu telah mengkaji penggunaan arsitektur CNN tunggal untuk klasifikasi kanker kulit. Pardede dan Putra (2020) menggunakan DenseNet121 dan mendapatkan nilai akurasi serta F-measure sebesar 0,94. Keunggulan arsitektur DenseNet terletak pada kemampuannya dalam memanfaatkan ulang fitur melalui koneksi antar lapisan sehingga dapat mengatasi masalah vanishing gradient. Penelitian lain oleh Salian dan Sawarkar (2022) menunjukkan bahwa EfficientNetB3 memiliki performa yang baik ketika dikombinasikan dengan teknik preprocessing dan Global Average Pooling, dengan capaian F1-Score sebesar 88%. Selain itu, Kousis et al. (2022) melakukan pengujian terhadap 11 arsitektur CNN menggunakan dataset HAM10000 dan menunjukkan bahwa DenseNet169 menjadi model terbaik dengan akurasi 92,25% dan F1-Score 93,27%, serta telah diimplementasikan dalam aplikasi mobile.

Walaupun model CNN tunggal memberikan hasil yang baik, pendekatan ini masih memiliki keterbatasan dalam menangani variasi data yang kompleks, terutama pada dataset dengan distribusi kelas yang tidak seimbang. Pada dataset HAM10000, sebagian besar data didominasi oleh kelas tertentu, sedangkan kelas lainnya memiliki jumlah yang jauh lebih sedikit. Kondisi ini menyebabkan metrik akurasi menjadi kurang representatif karena model cenderung mengalami bias dan lebih sering memprediksi ke arah kelas mayoritas (Kousis et al., 2022). Untuk mengatasi permasalahan tersebut, dapat diterapkan teknik seperti augmentasi citra yang bertujuan meningkatkan variasi data secara sintetis melalui modifikasi citra asli dan memperkaya sampel pada kelas minoritas seperti melanoma (Salian & Sawarkar, 2022). Eksperimen menunjukkan bahwa model dasar tanpa augmentasi hanya memperoleh akurasi sebesar 56,97%, dan berhasil meningkat secara signifikan menjadi 88,13% setelah teknik augmentasi diterapkan (Salian & Sawarkar, 2022).

Penelitian terbaru mulai mengarah pada pendekatan ensemble untuk meningkatkan kinerja klasifikasi, dengan menggabungkan beberapa arsitektur CNN agar dapat saling melengkapi dalam mengenali pola data yang kompleks. Pendekatan ini dilaporkan dapat meningkatkan performa dibandingkan penggunaan model tunggal, terutama pada kasus klasifikasi citra medis yang memiliki variasi visual dengan kompleksitas tinggi. Berdasarkan survei oleh Kadeem dan Mosa (2024), kombinasi beberapa model Convolutional Neural Network (CNN) dengan teknik optimasi terbukti dapat meningkatkan akurasi dan efisiensi dalam deteksi melanoma dengan tingkat akurasi mencapai 97,82% pada pengujian menggunakan dataset HAM10000 dan 99,46% secara umum. Hal ini menunjukkan bahwa penggabungan model dengan strategi optimasi dapat menghasilkan sistem yang lebih stabil dan andal dalam menangani variasi citra lesi yang kompleks. Meskipun capaian akurasi telah mencapai angka yang sangat baik, literatur mengidentifikasi adanya keterbatasan dalam aspek implementasi. Kendala utama terletak pada tingginya beban komputasi yang membatasi efisiensi model pada perangkat dengan sumber daya terbatas, serta karakteristik prediksi kecerdasan buatan yang bersifat tertutup. Hal ini menyebabkan rendahnya tingkat pemahaman informasi bagi pengguna mengenai alasan di balik hasil deteksi, yang menjadi hambatan dalam pengambilan keputusan.

Penggunaan beberapa arsitektur CNN yang memiliki karakteristik ekstraksi fitur yang berbeda memberikan keuntungan tersendiri. Arsitektur DenseNet121, sebagai contoh, dapat memperkuat aliran informasi fitur dan penggunaan ulang fitur melalui koneksi antar lapisan (dense blocks), sehingga dapat mengurangi jumlah parameter dan mengatasi permasalahan degradasi gradien (Pardede & Putra, 2020). Di sisi lain, EfficientNetB0 memiliki keunggulan dalam efisiensi komputasi melalui pendekatan compound scaling yang menyeimbangkan kedalaman jaringan, lebar, dan resolusi citra secara proporsional (Tan & Le, 2019). Sementara itu, ResNet50 dirancang untuk menjaga stabilitas pembelajaran pada jaringan yang dalam melalui mekanisme residual learning (He et al., 2016). Penggabungan beberapa model tersebut memungkinkan proses ekstraksi fitur dilakukan secara lebih menyeluruh untuk mengatasi keterbatasan masing-masing model tunggal.

Proses penentuan bobot pada model ensemble adalah faktor penting yang akan menentukan kualitas hasil prediksi akhir. Beberapa pendekatan optimasi metaheuristik seperti Algoritma Genetik sudah digunakan untuk mencari kombinasi bobot yang optimal pada model ensemble, bobot tersebut merepresentasikan kontribusi masing-masing model dalam proses penggabungan prediksi. Penelitian oleh Balasundaram et al. (2024) menunjukkan bahwa metode ensemble yang bobot tiap modelnya dioptimasi menggunakan Algoritma Genetik dapat mencapai akurasi hingga 91,73% pada dataset HAM10000, serta mengungguli metode penggabungan biasa. Temuan ini menunjukkan bahwa peran Algoritma Genetik cukup signifikan dalam meningkatkan performa sistem deteksi. Dalam konteks dataset medis dengan distribusi kelas yang tidak seimbang (imbalanced) seperti HAM10000 yang didominasi oleh kelas nevus (Kousis et al., 2022), performa model tidak cukup hanya diukur dengan akurasi. Penggunaan F1-Score sebagai metrik evaluasi dianggap lebih representatif karena dapat menyeimbangkan nilai precision dan recall. Dengan demikian, evaluasi menggunakan F1-Score dapat memberikan gambaran performa model yang lebih objektif dan adil terhadap seluruh kelas, serta meminimalkan risiko bias pada kelas mayoritas.

Penyajian hasil prediksi dalam bentuk tingkat risiko adalah aspek penting dalam pengembangan sistem deteksi melanoma, terutama bagi pengguna awam. Informasi yang disajikan dalam bentuk angka mentah seringkali sulit dipahami dan dapat menimbulkan kesalahpahaman pada interpretasi hasil. Diperlukan penyajian yang lebih komunikatif agar hasil prediksi dapat dipahami secara jelas. Penyajian dalam bentuk tingkat risiko, seperti risiko rendah, sedang, dan tinggi, dapat membantu pengguna memahami kondisi yang ditunjukkan sistem. Pendekatan ini mendukung fungsi sistem sebagai sarana deteksi dini dan peningkatan kesadaran, tanpa menggantikan peran diagnosis medis. Kousis et al. (2022) mengatakan bahwa sistem berbasis kecerdasan buatan dalam kesehatan perlu dirancang agar dapat memberikan informasi yang mudah dipahami pengguna.

Berdasarkan permasalahan yang telah dijelaskan sebelumnya, penelitian ini bertujuan untuk membangun sistem deteksi tingkat risiko melanoma berbasis kecerdasan buatan yang dapat diandalkan dan mudah diakses. Sistem yang dikembangkan memanfaatkan pendekatan Weighted Ensemble CNN dengan menggabungkan beberapa arsitektur model. Penggabungan model dengan metode rata-rata (simple averaging) cenderung menghasilkan performa prediksi yang akan kurang optimal karena pendekatan tersebut memperlakukan tiap arsitektur model dengan setara. Padahal masing-masing model pada dasarnya memiliki kapasitas pembelajaran dan ekstraksi fitur yang berbeda-beda (Balasundaram et al., 2024). Tanpa pembobotan yang ideal, model yang memiliki tingkat kesalahan lebih tinggi akan memberikan pengaruh yang sama besar dengan model yang paling akurat, sehingga menutup potensi akurasi maksimal dari sistem ensemble tersebut. Kemudian, pencarian kombinasi bobot secara manual atau brute-force akan memakan waktu dan daya komputasi yang sangat tidak efisien karena banyaknya kemungkinan kombinasi. Oleh sebab itu, penelitian ini menerapkan metode Algoritma Genetik untuk mencari kombinasi bobot secara heuristik, sehingga mampu menemukan parameter optimal untuk memaksimalkan metrik F1-Score dengan beban komputasi yang jauh lebih efisien. Selain itu, untuk memastikan aplikasi dapat berjalan lancar pada berbagai spesifikasi perangkat mobile, arsitektur client-server digunakan untuk menempatkan proses komputasi model yang berat pada sisi backend. Sebagai hasil akhirnya sistem akan memberikan output berupa tingkatan risiko. Dengan pendekatan ini, hasil deteksi menjadi lebih komunikatif dan mudah dipahami oleh masyarakat sebagai sarana deteksi dini yang praktis.

#### 1.2 Rumusan Masalah
Rumusan masalah dari pembuatan tugas akhir ini adalah “Bagaimana membuat sistem pada smartphone yang dapat mendeteksi melanoma dan menampilkan tingkatan risiko?”

#### 1.3 Tujuan
Tujuan dari tugas akhir ini adalah membangun sistem deteksi kanker kulit melanoma berbasis citra menggunakan metode Weighted Ensemble CNN yang dioptimasi dengan Algoritma Genetik dalam bentuk aplikasi hybrid mobile yang menampilkan output tingkat risiko bagi pengguna.

#### 1.4 Manfaat
Manfaat dari pembuatan sistem deteksi melanoma ini adalah sebagai berikut:
- **Manfaat bagi masyarakat:**
  - Menyediakan aplikasi deteksi dini berbasis hybrid mobile yang praktis untuk meningkatkan kesadaran masyarakat terhadap kondisi kulit.
  - Membantu masyarakat dalam melakukan deteksi dini melanoma serta pengecekan kondisi kulit secara mandiri.
  - Memberikan gambaran tingkat risiko melanoma kepada masyarakat.
- **Manfaat bagi peneliti:**
  - Menyediakan referensi bagi peneliti lain yang ingin mengembangkan optimasi model ensemble atau sistem deteksi penyakit berbasis citra medis nantinya.
  - Mendukung pengembangan metode optimasi metaheuristik untuk meningkatkan kinerja model ensemble pada pemrosesan citra medis.

#### 1.5 Ruang Lingkup
Ruang lingkup dari sistem ini adalah:
1. Penelitian ini menggunakan citra dermoskopi sebagai data latih.
2. Dataset yang digunakan adalah HAM10000 yang diperoleh dari Kaggle, berisi total 10015 citra dermoskopi yang terbagi dalam 7 kelas, yaitu: Nevus (6705), Melanoma (1113), BKL (1099), BCC (514), AKIEC (327), VASC (142), dan DF (115).
3. Sistem berbasis hybrid mobile menggunakan framework Ionic Angular dengan backend FastAPI.
4. Input pengguna memiliki standar kualitas tertentu untuk mencegah kesalahan prediksi akibat kualitas foto yang buruk dari kamera mobile. Foto dengan pencahayaan gelap, beresolusi sangat rendah, atau terlalu buram akan ditolak secara otomatis dan pengguna akan diminta untuk mengambil ulang foto.
5. Sistem menyajikan tiga tingkatan risiko kepada pengguna.
6. Sistem ini dirancang sebagai sistem deteksi dini bukan sebagai diagnosis medis.
7. Penelitian ini tidak akan membahas aspek medis dari melanoma dan hanya berfokus pada aspek teknis pembuatan model serta implementasinya.

#### 1.6 Metodologi Penelitian
Metodologi dalam penelitian ini disusun melalui beberapa tahapan berikut:
- **Persiapan:** Pada tahap ini dilakukan studi literatur terkait klasifikasi lesi kulit dan model CNN yang digunakan (ResNet50, DenseNet121, dan EfficientNetB0), serta pencarian dataset dermoskopi lesi kulit yang bersifat publik untuk penelitian ini.
- **Analisis:** Pada tahap ini dilakukan analisis kebutuhan sistem untuk menentukan fitur dan fungsi yang akan diimplementasikan, serta analisis dataset yang telah diperoleh untuk memahami karakteristik, jumlah, dan kualitas data.
- **Desain:** Tahap ini berfokus pada perancangan sistem secara keseluruhan yang meliputi arsitektur model, alur sistem, serta antarmuka pengguna aplikasi. Pada tahap ini dirancang model CNN yang akan digunakan dalam skema weighted ensemble, serta metode optimasi menggunakan Algoritma Genetik. Selain itu, dirancang alur proses sistem mulai dari input citra, tahap preprocessing, hingga proses prediksi dan penyajian hasil. Sementara pada sisi aplikasi, dilakukan perancangan antarmuka pengguna (UI/UX) yang disesuaikan dengan kebutuhan pengguna agar mudah digunakan dan informatif.
- **Implementasi:** Pada tahap ini dilakukan pembangunan sistem berdasarkan rancangan yang telah dibuat. Dataset yang telah dikumpulkan akan diproses melalui tahap preprocessing sebelum digunakan untuk melatih model. Selanjutnya dilakukan implementasi model CNN dalam skema weighted ensemble serta penerapan algoritma optimasi menggunakan Algoritma Genetik untuk menentukan bobot terbaik dari setiap model. Dilakukan pengembangan aplikasi berbasis hybrid mobile yang terintegrasi dengan sistem deteksi yang telah dibangun.
- **Uji Coba dan Evaluasi:** Pada tahap ini dilakukan pengujian terhadap sistem yang telah dikembangkan untuk mengetahui kinerja model dan aplikasi. Pengujian dilakukan menggunakan metrik evaluasi akurasi, precision, recall, dan F1-score untuk menilai performa klasifikasi. Dilakukan pengujian aplikasi untuk memastikan bahwa sistem berjalan dengan baik, mulai dari proses input citra hingga penyajian hasil deteksi.
- **Penyusunan Laporan:** Pada tahap ini dilakukan penyusunan laporan Tugas Akhir yang mendokumentasikan seluruh proses pengembangan sistem. Laporan mencakup tahapan persiapan, analisis, perancangan, implementasi, serta hasil pengujian dan evaluasi yang telah dilakukan.

#### 1.7 Sistematika Penulisan
Struktur penulisan tugas akhir ini dibuat untuk memberikan pemahaman yang jelas tentang isi tugas akhir, yang dijabarkan sebagai berikut:
- **Bab 1: Pendahuluan:** Membahas latar belakang penelitian, rumusan masalah, tujuan, manfaat, ruang lingkup, metodologi penelitian, dan sistematika penulisan laporan.
- **Bab 2: Landasan Teori:** Membahas terkait landasan teori yang mendukung penelitian ini, yaitu sistem deteksi, preprocessing, arsitektur CNN, ensemble model, optimasi dengan algoritma genetik, dan metode evaluasi model.
- **Bab 3: Analisis Sistem:** Membahas mengenai analisis kebutuhan sistem melalui kuesioner dan wawancara dengan masyarakat untuk mengidentifikasi kelayakan fitur.
- **Bab 4: Desain Sistem:** Membahas mengenai desain preprocessing, model, dan user interface dari sistem yang dibuat.
- **Bab 5: Implementasi Sistem:** Membahas implementasi dari ensemble model yang digunakan untuk mendeteksi melanoma dengan gabungan 3 arsitektur CNN (ResNet50, DenseNet121, EfficientNetB0).
- **Bab 6: Evaluasi Sistem:** Membahas hasil dari uji coba dan evaluasi terhadap sistem yang telah dibuat untuk menentukan konfigurasi paling optimal.
- **Bab 7: Kesimpulan dan Saran:** Membahas kesimpulan akhir dari penelitian serta memberikan saran yang dapat digunakan untuk pengembangan pada penelitian selanjutnya.

---

### BAB 2 LANDASAN TEORI
Bab 2 membahas mengenai landasan teori yang menjadi fondasi dari sistem yang dikembangkan pada tugas akhir ini.

#### 2.1 Kanker Kulit Melanoma
Kanker kulit merupakan kelompok penyakit keganasan yang ditandai oleh pertumbuhan sel-sel kulit secara abnormal dan tidak terkendali akibat kerusakan pada materi genetik sel. Berdasarkan laporan resmi World Health Organization (WHO) pada tahun 2022, terdapat lebih dari 1,5 juta kasus baru kanker kulit yang tercatat secara global (World Health Organization, 2022).

Di antara berbagai klasifikasi kanker kulit, melanoma merupakan jenis yang paling agresif dan berbahaya. Melanoma berasal dari transformasi maligna sel melanosit, yaitu sel yang secara alami bertugas memproduksi melanin sebagai pigmen pelindung kulit dari paparan sinar matahari. Sel melanoma memiliki kemampuan bermetastasis yang sangat tinggi melalui sistem limfatik maupun aliran darah menuju organ-organ vital seperti paru-paru, hati, dan otak dalam waktu yang singkat. Karakteristik inilah yang menjadikan melanoma sebagai penyebab kematian terbanyak di antara seluruh jenis kanker kulit, meskipun secara proporsi jumlah kasusnya relatif lebih kecil dibanding jenis kanker kulit non-melanoma (Kadeem & Mosa, 2024).

Secara klinis, diagnosis melanoma pada umumnya dilakukan oleh dokter spesialis dermatologi melalui dua pendekatan utama. Pertama adalah pemeriksaan visual fisik menggunakan kriteria ABCDE (*Asymmetry, Border, Color, Diameter, Evolution*) sebagai panduan identifikasi karakteristik lesi yang mencurigakan. Kedua adalah pemeriksaan dermoskopi, yaitu teknik pencitraan non-invasif yang memanfaatkan cahaya terpolarisasi untuk mengobservasi struktur morfologis kulit pada kedalaman epidermis dan dermis bagian atas yang tidak dapat terlihat oleh mata telanjang (Abbas et al., 2011).

#### 2.2 Convolutional Neural Network (CNN)
Deep Learning adalah cabang dari Machine Learning yang menggunakan arsitektur jaringan saraf tiruan berlapis dalam (*deep neural network*) untuk mempelajari representasi fitur data secara hierarkis dari lapisan abstraksi rendah ke tinggi tanpa rekayasa fitur manual.

Convolutional Neural Network (CNN) adalah arsitektur deep learning yang secara khusus dirancang untuk memproses data yang memiliki struktur topologi grid, dengan aplikasi utama pada citra dua dimensi. Keunggulan fundamental CNN dibandingkan jaringan saraf tiruan biasa terletak pada tiga properti utama: *local connectivity*, *parameter sharing*, dan *translational invariance* (Li et al., 2022).

Arsitektur CNN secara umum terdiri dari susunan lapisan-lapisan berikut:
- **Convolutional Layer:** Komponen komputasi inti melaksanakan operasi konvolusi diskret antara tensor citra input dengan sekumpulan filter (kernel) kecil (3×3 atau 5×5) untuk menghasilkan *feature map*.
- **Activation Function (ReLU):** Fungsi aktivasi Rectified Linear Unit, f(x) = max(0, x), memperkenalkan properti non-linearitas dan efektif mengatasi masalah *vanishing gradient* (Ioffe & Szegedy, 2015).
- **Pooling Layer:** Menjalankan fungsi *downsampling* atau pengurangan dimensi spasial dari feature map (misal lewat *Max Pooling*) untuk menghemat beban komputasi.
- **Batch Normalization:** Menormalkan aktivasi pada setiap mini-batch untuk mempercepat konvergensi dan bertindak sebagai regularizer implisit.
- **Fully Connected Layer dan Softmax Layer:** Mengubah skor mentah (logits) dari feature map linear menjadi distribusi probabilitas kelas kumulatif bernilai total 1.

##### 2.2.1 Transfer Learning
Transfer learning adalah paradigma dalam machine learning yang memanfaatkan representasi pengetahuan yang telah dipelajari oleh sebuah model dari domain atau tugas sumber (*source domain/task*, seperti ImageNet) untuk mempercepat dan meningkatkan proses pembelajaran pada domain target yang baru. Dalam citra dermoskopi medis, transfer learning sangat berharga untuk mengatasi keterbatasan jumlah dataset terlabeli serta menghemat waktu komputasi pelatihan dari nol (Tan & Le, 2019).

##### 2.2.2 Arsitektur Pretrained CNN
1. **ResNet50:** Diperkenalkan oleh He et al. (2016) menggunakan inovasi *skip connection* atau *shortcut connection* untuk melewatkan gradien langsung antar blok tanpa terdegradasi. Menggunakan struktur *bottleneck block* (konvolusi 1x1, 3x3, 1x1) untuk menyeimbangkan kedalaman jaringan sebanyak 50 lapisan berbobot.
2. **DenseNet121:** Diperkenalkan oleh Huang et al. (2017) menggunakan konsep konektivitas padat (*dense connectivity*) di mana setiap lapisan menerima *feature map* gabungan (*concatenation*) dari seluruh lapisan sebelumnya dalam blok yang sama. Efektif memperkuat aliran informasi dan meminimalkan redundansi parameter (Pardede & Putra, 2020).
3. **EfficientNetB0:** Diperkenalkan oleh Tan dan Le (2019) dengan pendekatan penskalaan terpadu (*compound scaling*) yang menyeimbangkan kedalaman, lebar, dan resolusi input secara proporsional. Menggunakan blok dasar *Mobile Inverted Bottleneck Convolution* (MBConv) dengan mekanisme *Squeeze-and-Excitation* (SE) untuk komputasi mobile yang sangat efisien.

#### 2.3 Weighted Ensemble CNN
Ensemble learning menggabungkan prediksi dari beberapa model individual untuk menghasilkan luaran keputusan akhir yang lebih akurat dan tangguh (*robust*). Dalam penelitian ini, tiga model dengan bias induktif berbeda (ResNet50, DenseNet121, dan EfficientNetB0) dijalankan bersama secara paralel. 

Pendekatan *Weighted Ensemble* memberikan bobot (*weight*) pengali yang bervariasi pada representasi skor probabilitas tiap model berdasarkan kompetensi performanya, berbeda dari *simple averaging* yang membagi rata semua model secara identik (Balasundaram et al., 2024). Formula agregasi akhir didefinisikan sebagai:

$$P_{akhir} = (w_1 	imes P_1) + (w_2 	imes P_2) + (w_3 	imes P_3)$$

Dengan batasan mutlak: $w_1 + w_2 + w_3 = 1.0$ dan $w_i \ge 0$.

#### 2.4 Optimasi Algoritma Genetik
Algoritma Genetik (GA) adalah metode optimasi metaheuristik berbasis populasi yang meniru proses seleksi alam dan evolusi biologis untuk menavigasi ruang pencarian solusi yang kompleks dan multidimensional (Katoch et al., 2021). GA digunakan untuk mencari kombinasi bobot $[w_1, w_2, w_3]$ optimal demi memaksimalkan nilai *macro-averaged F1-Score* pada data validasi.

Komponen siklus evolusinya meliputi:
- **Representasi Kromosom:** Berupa array satu dimensi berisi tiga nilai bertipe *floating-point* yang merepresentasikan bobot masing-masing arsitektur CNN.
- **Inisialisasi Populasi:** Membangkitkan sejumlah $N$ kromosom acak yang telah dinormalisasi agar memenuhi total penjumlahan bernilai 1.0.
- **Fitness Function:** Evaluasi performa empiris model ensemble menggunakan kromosom tersebut pada data validasi. Nilai *F1-Score* yang tinggi ekuivalen dengan *fitness value* yang tinggi.
- **Seleksi:** Memilih individu induk (*parents*) terbaik berdasarkan nilai fitness tertinggi agar sifat unggulnya dapat diwariskan ke generasi selanjutnya.
- **Persilangan (Crossover):** Menggunakan *arithmetic crossover* dengan parameter rasio $lpha = 0.5$ untuk menghasilkan kromosom anak (*offspring*) baru melalui kombinasi linear.
- **Mutasi:** Pergeseran nilai gen secara stokastik terkontrol (*mutation rate*) untuk menjaga diversitas populasi dan mencegah jebakan *local optima*.
- **Kondisi Berhenti:** Proses evolusi disudahi ketika mencapai batas generasi ke-100 atau performa fitness telah konvergen.

#### 2.5 Dataset dan Preprocessing Citra
##### 2.5.1 Dataset HAM10000
Dataset HAM10000 dikumpulkan oleh Tschandl et al. (2018) berisi 10.015 citra dermoskopi lesi kulit berwarna yang terbagi ke dalam 7 kelas dengan tingkat ketidakseimbangan kelas (*severe class imbalance*) yang sangat tinggi:

| Nama Kelas | Singkatan | Jumlah Citra |
| :--- | :--- | :--- |
| Melanocytic Nevi (Tahi Lalat Jinak) | NV / Nevus | 6705 |
| Melanoma | MEL | 1113 |
| Benign Keratosis-like Lesions | BKL | 1099 |
| Basal Cell Carcinoma | BCC | 514 |
| Actinic Keratoses | AKIEC | 327 |
| Vascular Lesions | VASC | 142 |
| Dermatofibroma | DF | 115 |
| **Total** | **—** | **10.015** |

##### 2.5.2 Segmentasi Citra dengan Otsu’s Thresholding
Segmentasi otomatis menggunakan metode *Otsu's Thresholding* pada citra keabuan (*grayscale*) untuk memisahkan area lesi (*foreground*) dengan kulit sehat (*background*). Metode ini secara adaptif meminimalkan varian dalam kelas (*within-class variance*) atau memaksimalkan varian antar kelas (*between-class variance*) melalui persamaan:

$$\sigma^2_B(T) = \omega_1(T) \cdot \omega_2(T) \cdot [\mu_1(T) - \mu_2(T)]^2$$

Berdasarkan citra biner luaran segmentasi, ditentukan koordinat batas ekstrim sumbu X ($x_{min}, x_{max}$) dan Y ($y_{min}, y_{max}$) untuk melakukan pemotongan (*cropping*) fokus lesi (Nagaoka, 2025).

##### 2.5.3 Restorasi Citra (Hair Removal)
Untuk mengeliminasi gangguan artefak rambut, diterapkan filter morfologi *Blackhat* yang didefinisikan sebagai selisih operasi *closing* dengan citra asli:

$$Blackhat(f) = (f ullet b) - f$$

Hasil filter berupa biner *hair mask* kemudian direstorasi menggunakan algoritma `INPAINT_TELEA` yang memanfaatkan *Fast Marching Method* (FMM) untuk mengisi piksel rambut dengan nilai rata-rata estimasi piksel kulit sehat di lingkungan sekitarnya (Telea, 2004; Salian & Sawarkar, 2022).

##### 2.5.4 Augmentasi Citra
Untuk menyeimbangkan distribusi data saat pelatihan, kelas minoritas diperbanyak hingga setara dengan kelas mayoritas (6.705 citra) menggunakan kombinasi transformasi acak: *Rotation* ($-45^\circ$ hingga $+45^\circ$), *Horizontal Flip*, dan *Vertical Flip* (Salian & Sawarkar, 2022). Seluruh dataset dibagi dengan rasio 70:15:15 untuk *Train set*, *Validation set*, dan *Testing set*.

#### 2.6 Penentuan Tingkat Risiko
Untuk memberikan representasi informasi yang komunikatif bagi masyarakat awam, skor probabilitas kumulatif akhir dari Weighted Ensemble dipetakan secara deterministik (*rule-based risk categorization*) menjadi 3 tingkatan kategori risiko (Kousis et al., 2022):

| Tingkat Risiko | Rentang Probabilitas | Keterangan |
| :--- | :--- | :--- |
| **Rendah** | 0.00 – 0.39 | Karakteristik visual lesi tidak menunjukkan indikasi yang signifikan ke arah melanoma. |
| **Sedang** | 0.40 – 0.69 | Terdapat beberapa karakteristik yang perlu diwaspadai. Disarankan berkonsultasi dengan tenaga medis. |
| **Tinggi** | 0.70 – 1.00 | Model mendeteksi karakteristik visual yang kuat mengindikasikan melanoma. Sangat disarankan segera berkonsultasi dengan dokter spesialis. |

#### 2.7 Evaluasi
Pengukuran kualitas model dilakukan menggunakan skema *Confusion Matrix* melalui ekstraksi nilai *True Positive* (TP), *True Negative* (TN), *False Positive* (FP), dan *False Negative* (FN) untuk kalkulasi beberapa metrik standar evaluasi:

- $Accuracy = rac{TP + TN}{TP + TN + FP + FN}$
- $Precision = rac{TP}{TP + FP}$
- $Recall = rac{TP}{TP + FN}$
- $F1	ext{-}Score = 2 	imes rac{Precision 	imes Recall}{Precision + Recall}$

*Macro-averaged F1-Score* dijadikan acuan performa utama karena memberikan bobot penilaian yang adil di setiap kelas tanpa terpengaruh dominasi ukuran data (Kousis et al., 2022).

#### 2.8 Framework dan Library
- **Ionic Angular:** Framework pengembangan *hybrid mobile* lintas platform pada sisi *front-end client* untuk membangun antarmuka UI/UX yang responsif.
- **FastAPI:** Framework backend berbasis Python berkinerja tinggi dengan dukungan eksekusi asinkron (*asynchronous*) untuk mengekspos REST API model.
- **TensorFlow / Keras:** Framework utama pemuatan arsitektur pretrained, proses *fine-tuning*, serta kalkulasi probabilitas prediksi model.
- **OpenCV:** Pustaka manipulasi citra digital untuk menangani ekstraksi *Otsu's Thresholding*, filter *Blackhat*, dan inpainting `TELEA`.
- **NumPy:** Pengolah komputasi array multidimensi (tensor) untuk memfasilitasi operasi aljabar linier pembobotan ensemble dan struktur genetik GA.
- **Scikit-Learn (Sklearn):** Digunakan dalam pembagian proporsi dataset serta kalkulasi metrik evaluasi *confusion matrix*.

---

### BAB 3 ANALISIS SISTEM
#### 3.1 Kondisi Saat Ini
Berisi rencana pengumpulan data primer melalui wawancara terstruktur bersama dokter spesialis dermatologi untuk memetakan alur penanganan klinis melanoma, penentuan parameter urgensi risiko, serta mengukur tingkat kesadaran (*awareness*) masyarakat saat ini.

#### 3.2 Analisis Sistem Sejenis
Melakukan komparasi fungsionalitas fitur dan performa dari beberapa aplikasi sejenis yang telah beredar di publik guna menemukan *gap* pengembangan.

#### 3.3 Analisis Permasalahan
Mengidentifikasi hambatan komputasi mobile lokal, variasi kualitas kamera, serta keterbatasan visual manusia dalam membedakan lesi melanoma dengan nevus jinak.

#### 3.4 Analisis Kebutuhan Sistem
Merumuskan spesifikasi fungsional dan non-fungsional sistem (seperti batas retensi kualitas gambar input, kecepatan respons API server, dan kesederhanaan UI).

---

## BAGIAN 2: PROPOSAL TUGAS AKHIR
*Proposal TA_Alexander F. J. Lapod_160423232.docx*

*(Isi materi, latar belakang, rumusan masalah, tujuan, manfaat, ruang lingkup, metodologi penelitian, serta simulasi matematis perhitungan iterasi seleksi, persilangan aritmatika, mutasi genetik, dan agregasi skor Weighted Ensemble pada Proposal ini selaras dan berkorespondensi penuh dengan struktur teks yang tercantum pada BAB 1 dan BAB 2 Laporan Tugas Akhir di atas).*
