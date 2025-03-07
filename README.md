# **Used Sailboat Price Prediction**

## **Project Overview**
Used sailboats are expensive, often listed in the range of $100,000 and sometimes in the millions. Various factors influence their pricing, including boat specifications and regional economic conditions. This project establishes a **Random Forest - Multiple Linear Regression (RF-MLR) composite model** to analyze the relationship between listing prices and influencing factors, identify potential regional effects, and forecast the Hong Kong market.

## **Data Preprocessing**
- **Data Collection**: 15 influencing factors (e.g., production year) were selected based on hull characteristics and regional economic indicators. The dataset was expanded using Python web scraping.
- **Feature Selection**: Analyzed the correlation between the 15 factors and listing price, removing four factors (e.g., headroom height) due to low correlation.
- **Data Cleaning**: Conducted outlier analysis and handled missing values for the remaining 11 factors.

## **Model Development**
1. **Random Forest (RF) Training**:
   - Trained separate models for monohulled sailboats and catamarans.
   - Selected the top 1/3 most important variables for model refinement.
2. **Multiple Linear Regression (MLR) Fitting**:
   - Mapped critical features to listing price using MLR.
   - Converted categorical variables (e.g., manufacturer’s brand) into dummy variables for regression analysis.
   - Derived mathematical expressions for sailboat listing prices.
3. **Model Performance Evaluation**:
   - The variation with error value within 10% accounted for **96% (monohulls)** and **94% (catamarans)**, demonstrating strong predictive performance.

## **Regional Effects Analysis**
- Compared the **average listing prices** of sailboats across three geographical regions: **Caribbean, Europe, and USA**.
- **Findings**:
  - **USA > Total ≈ Europe > Caribbean** in terms of average listing price.
  - The RF-MLR model showed that:
    - **Monohull prices** correlate positively with **regional per capita consumption expenditure (2020)**.
    - **Catamaran prices** correlate positively with **regional average tourism income (last 5 years)**.
  - Some variations did not align with regional trends, leading to further analysis of **statistical and practical significance**.

## **Hong Kong Market Forecasting**
- Introduced an **actual listing price correction** in the MLR models.
- Developed a new **Regional Effect Evaluation Function (RE)**.
- Compared second-hand sailboat data in **Hong Kong** with the original dataset:
  - Applied RE function to overlapping data points.
  - Found that **Hong Kong sailboat prices** were **higher** than in the other three regions.
  - The **regional effects of monohulls and catamarans remained consistent**.

## **Keywords**
- Random Forest
- Multiple Linear Regression
- Pricing Model
- Sailboat

## **Repository Structure**
```
📂 Used_boat_price_prediction
├── 📂 boat              # boat data
├── 📂 input             # Trained model files and results
├── 📂 output
├── 📂 random_forest     # Python scripts for data preprocessing and model training
├── README.md             # Project documentation
```

## **How to Use**
1. Clone this repository:
   ```sh
   git clone https://github.com/Ruihan1124/Used_boat_price_prediction.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the main analysis script:
   ```sh
   python scripts/train_model.py
   ```

## **License**
This project is released under the [MIT License](LICENSE).

---

📌 **Author:** Ruihan1124  
📌 **Contact:** [GitHub](https://github.com/Ruihan1124)
