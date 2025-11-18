#  Chronic Kidney Disease (CKD) Surunkali Buyrak Yetishmovchiligini labarator tahlillari asosida aniqlovchi AI model. 

🇺🇿Foydalanish qo'llanmasi eng pastki qismda!

A machine learning project for early detection of **Chronic Kidney Disease (CKD)** based on routine clinical and laboratory measurements.  
The model is implemented as a **clean sklearn Pipeline**, making it easy to use directly from the terminal, in Streamlit, or inside any Python environment.

This project is part of my portfolio in **Biotechnologies and Applied Artificial Intelligence for Health**  
(MSc, University of Pisa).

---

## 📌 Project Overview

Chronic Kidney Disease (CKD) is a progressive condition that often remains asymptomatic until advanced stages.  
Early detection can significantly improve patient outcomes.

In this project, I built a light and interpretable ML pipeline that predicts CKD risk using only **8 common clinical features**:

- `age` – patient age  
- `bp` – blood pressure  
- `bgr` – random blood glucose  
- `bu` – blood urea  
- `sc` – serum creatinine  
- `hemo` – hemoglobin  
- `htn` – hypertension (yes/no)  
- `dm` – diabetes mellitus (yes/no)

All preprocessing (imputation + categorical encoding) and the classifier are wrapped into a single sklearn Pipeline for portability.

---

## 🧠 Model Architecture

The pipeline consists of:

- `SimpleImputer` — filling missing numerical and categorical values  
- `OneHotEncoder` — encoding `htn` and `dm`  
- `RandomForestClassifier` — final model  
- `Pipeline` — combining all steps together

The model was trained using the **UCI CKD dataset**.

---

> ⚠️ Note: The CKD dataset is small and imbalanced, so results should be interpreted with caution.

## 📊 Model Visualizations

### Confusion Matrix
<img src="/conf_matrix.png" width="420"/>

### ROC Curve
<img src="/ROC_curve.png" width="420"/>

### Example Prediction (Single Patient)
<img src="/example_test.png" width="420"/>


# 🇺🇿Foydalish (O'zbek tilida)
## Dastlab yuqoridan pipeline faylini yuklab oling, va  

Pythonda quyidagilarni yozib app.py nomi bilan saqlab oling, undan so'ng terminalingizda shu kodni kiriting: streamlit run app.py
Local hostda yangi display ochiladi, u foydalanishga juda qulay va siz bemalol bemorlarning malumotlarini kiritib tahlil qilishingiz mumkin!

```python

import streamlit as st
import pandas as pd
import joblib

pipeline_path = "" #qo'shtirnoq ichiga shu yerdan yuklab olgan pipeline fayli joylashgan path ni kiriting masalan:...downloads/ckd_rf_simple_pipeline.pkl
pipeline = joblib.load(pipeline_path)

st.title("Surunkali buyrak yetishmovchiligi (Chronic Kidney Disease) – ni tahlillarga asosan baholovchi model (@byshukrullo)")
st.write("Quyidagi laborator va klinik ko'rsatkichlar asosida CKD xavfini baholaydi.")

st.subheader(" Bemor labarator ma'lumotlarini kiriting:")

age  = st.number_input("Yosh (age)",  value=25)
bp   = st.number_input("Qon bosimi (bp)", value=120)
bgr  = st.number_input("Blood Glucose Random (bgr)", value=100)
bu   = st.number_input("Blood Urea (bu)", value=30)
sc   = st.number_input("Serum Creatinine (sc)", value=0.8)
hemo = st.number_input("Hemoglobin (hemo)", value=14.0)

htn = st.selectbox("Hypertension (htn)", ["no", "yes"])
dm  = st.selectbox("Diabetes Mellitus (dm)", ["no", "yes"])

if st.button("CKD xavfini baholash"):
    data = {
        'age': age,
        'bp': bp,
        'bgr': bgr,
        'bu': bu,
        'sc': sc,
        'hemo': hemo,
        'htn': htn,
        'dm': dm
    }

    df_new = pd.DataFrame([data])

    proba = pipeline.predict_proba(df_new)[0, 1]
    pred = pipeline.predict(df_new)[0]

    if pred == 1:
        st.error(f"CKD xavfi YUQORI. Probability = {proba:.2f}")
    else:
        st.success(f"CKD xavfi past. Probability = {proba:.2f}")

    st.caption("Model faqat ta’limiy va qo‘shimcha qaror qo‘llab-quvvatlash uchun. Yakuniy tashxisni shifokor qo‘yadi. Create by Shukrullo Foziljonov") 
