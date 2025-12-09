

# 🎵 Music Genre Classification (GMM + SVM Fusion)

This project classifies audio files into **4 music genres** using **MATLAB**.
It uses **MFCC features**, **Gaussian Mixture Models (GMM)**, **Support Vector Machines (SVM)**, and a **fusion method** that combines both models for higher accuracy.

---

## 🎧 Genres Classified

The script automatically detects **4 genre folders** from your dataset directory.

Common examples:

* Rock
* Pop
* Jazz
* Classical

> ✔ Folder names can be anything — the model learns from the first 4 subfolders it finds.

---

##  How the Model Works

### 1. Feature Extraction

For each audio file, the script extracts important features:

* **MFCCs** (captures timbre)
* **Spectral Centroid**
* **Spectral Bandwidth**
* **Spectral Rolloff**
* **Spectral Flatness**
* **Zero Crossing Rate**
* **RMS Energy**

These features convert the audio signal into a meaningful mathematical representation.

---

##  What is GMM? (Simple Explanation)

A **Gaussian Mixture Model** represents each genre using multiple bell-curve distributions.

How it works:

* Looks at **all MFCC frames** of each audio file
* Learns the statistical pattern of frames for each genre
* For a new song, it checks:
  **“Which genre’s pattern does this match the most?”**

**Strengths:**

* Excellent for modeling **frame-level sound variations**
* Works well with **lots of short audio segments**

---

##  What is SVM? (Simple Explanation)

A **Support Vector Machine** finds the best boundary between genres.

How it works:

* Each song is converted into **one feature vector**
* SVM learns to separate genres using these features
* Predicts the genre for each track

**Strengths:**

*  Strong at **global track-level classification**
*  Works well with well-designed feature vectors

---

## Why Fusion (GMM + SVM)?

Both models contribute differently:

| Model      | Strength                                       |
| ---------- | ---------------------------------------------- |
| **GMM**    | Understands detailed **frame-level** patterns  |
| **SVM**    | Understands **overall track-level** structure  |
| **Fusion** | Combines both for **best overall performance** |

The fusion method boosts accuracy by using both local and global perspectives of the audio.

