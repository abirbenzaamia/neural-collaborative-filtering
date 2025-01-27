# **FedMLP: Federated Recommendation System**

## **Overview**
FedRec proposes the federated learning versions of recommendation algorithms proposed by [He et al., 2017]. FedRec includes 3 federated algorithms: Federated Generalized Matrix Factorization (FedGMF), Federated Neural Collaborative Filtering (FedNCF) and Federated Neural Matrix Factorization (FedNeuMF). The repo contains necessary code to evaluate models' performance with different metrics and compare their performance to centralized versions.

---

## **Key Features**
- Centralized and Federated code for Collaborative filtering.
- Evaluation metrics like HR@10, NDCG@10, MAP@10 and MAR@10 for recommendation quality assessment
- Supports popular datasets like **MovieLens**, **Amazon**, and **Foursquare**.  
- Visualization tools to plot performance metrics.

---

## **Getting Started**

### **Prerequisites**
- Acess the directory ./federated for the federated learning models.
- Python 3.8 or higher
- Required Python libraries (install using `requirements.txt`):
  ```bash
  pip install -r federated/requirements.txt
