# **FedMLP: Federated Recommendation System**

## **Overview**
FedMLP implements a federated recommendation system using a neural adaptation of the matrix factorization algorithm proposed by [He et al., 2017]. This model combines linear and non-linear relationships to improve recommendation quality. Unlike centralized architectures, where sensitive user data is stored on a central server, this solution leverages **Federated Learning (FL)** to train models locally on user devices, ensuring data privacy.

The framework also supports federated analytics and evaluation metrics to analyze recommendation quality under privacy-preserving constraints.

---

## **Key Features**
- Neural adaptation of matrix factorization with support for **linear** and **non-linear relationships**.
- Privacy-preserving **Federated Learning (FL)** setup, where user data remains local, ensuring confidentiality.  
- Supports popular datasets like **MovieLens**, **Amazon**, and **Foursquare**.  
- Federated analytics using differential privacy (DP) for discovering popular items while maintaining privacy guarantees.  
- Evaluation metrics including **precision**, **recall**, and **F1-score** for recommendation quality assessment.  
- Visualization tools to plot performance metrics.

---

## **Getting Started**

### **Prerequisites**
- Python 3.8 or higher
- Required Python libraries (install using `requirements.txt`):
  ```bash
  pip install -r requirements.txt
