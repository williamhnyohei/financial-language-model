# **Financial-Language-Model**

**Financial-Language-Model** is a specialized Natural Language Processing (NLP) system designed to analyze stock market fluctuations from 2000 to the present. By leveraging historical data, economic events, key influencers, and financial formulas, it predicts and interprets stock price movements based on user-defined scenarios. This project is intended for educational and research purposes only.

---

## **Features**
- Analyze historical stock price fluctuations and identify key factors influencing the market.
- Predict potential stock price trends based on specific scenarios (e.g., changes in interest rates, economic events).
- Learn from datasets enriched with events, numerical data, and contextual explanations.
- Easily customizable for other datasets or financial domains.

---

## **Dataset**
The model uses historical stock data enriched with events, percentages, and key influencers.

Example data includes:
Historical prices (e.g., open, close, volume, etc.).
Major events (e.g., interest rate hikes, corporate earnings, geopolitical events).
Key influencers (e.g., Federal Reserve decisions, CEO announcements).
The primary dataset is located at data/stock_phrases.txt, but you can customize this with your own financial datasets.

---

## **Installation**

1. **Clone the Repository**:
  ```
   git clone https://github.com/yourusername/financial-language-model.git
   cd financial-language-model
  ```

2. Set Up Virtual Environment (optional but recommended):
  ```
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
  ```
3. Install Dependencies:
  ```
   pip install -r requirements.txt
  ```

---

## **Usage**

1. Training the Model
To train the model using your financial dataset:
  ```
   python src/train.py --data data/frases_acoes.txt --epochs 10 --batch_size 32
  ```
2. Running Inference
To predict or analyze a specific scenario:
  ```
   python src/inference.py --text "Interest rates are expected to rise by 0.5% next quarter."
  ```
3. Example Output
  ```
   Given the input scenario:
  "Interest rates are expected to rise by 0.5% next quarter."
  
  Model prediction:
  - Stocks in the technology sector are likely to drop by 2-4% due to increased borrowing costs.
  - Defensive sectors, such as utilities, may see moderate growth.
  ```

---

## **License**
This project is licensed under the MIT License, allowing for permissive use, modification, and distribution with attribution.

---

## **Disclaimer**
This project is intended for educational and research purposes only. It should not be used for financial or investment decisions.

