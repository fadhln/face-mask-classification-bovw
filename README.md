# Klasifikasi Citra Berbasis Bag-of-Visual-Words (BoVW)

*Repository* ini diadaptasi dari <a href="https://github.com/tobybreckon/python-bow-hog-object-detection">tobybreckon/python-bow-hog-object-detection</a> yang telah mengimplementasikan algoritma *Bag-of-Visual-Words* menggunakan Python dan OpenCV dengan memanfaatkan *classifier* SVM.

Dataset yang digunakan pada eksperimen ini berasal dari <a href="https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset">Face Mask Detection ~12K Images Dataset</a> yang disimpan pada website Kaggle.

---

## *Dependencies:*

Pustaka yang digunakan pada program ini adalah:

```
numpy==1.19.2
opencv-python>=3.3.0
tqdm==4.50.2
```

Kebutuhan tersebut telah terkompilasi pada `requirements.txt`, sehingga proses instalasi pustaka dapat dilakukan dengan:

```shell
pip install -r requirements.txt
```

---

## Cara menjalankan:

Program utama terdapat dalam `bovw-classification.ipynb` yang disajikan dalam bentuk *Python Notebook*. Untuk dapat menjalankan program, langkah-langkah yang harus dilakukan adalah

1. Unduh dataset dari <a href="https://www.kaggle.com/ashishjangra27/face-mask-12k-images-dataset">Face Mask Detection ~12K Images Dataset</a> dan simpan ke folder `dataset`.
2. Jalankan perintah `pip install -r requirements.txt` pada terminal.
3. Jalankan `bovw-classification.ipynb` untuk mendapatkan hasil klasifikasi.

---

## Tambahan informasi:

Parameter pelatihan dapat dimodifikasi pada file `params.py`. Parameter pelatihan antara lain nama kelas, metode ekstraksi fitur, *hyperparameter* SVM, *path* dataset, dan lain-lain.