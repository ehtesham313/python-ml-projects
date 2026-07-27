import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

def train_and_save_model():
    print("Training Email Spam Classification Model...")

    # Representative training dataset (Spam vs Ham messages)
    data = [
        # Spam samples
        ("WINNER!! As a valued network customer you have been selected to receive a $1000 cash prize!", "spam"),
        ("URGENT! You have won a 1 week FREE membership in our $100,000 Prize Jackpot! Call now to claim.", "spam"),
        ("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)", "spam"),
        ("Congratulations! You've been selected for a free $500 Walmart gift card. Click link now to claim.", "spam"),
        ("SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575.", "spam"),
        ("Claim your free iPhone 15 Pro Max today! Exclusive offer for selected users. Click here to confirm address.", "spam"),
        ("Hot Singles in your area are looking for chat partners! Click now to join for free.", "spam"),
        ("Urgent notice: Your bank account has been locked. Verify your credentials immediately at http://fakebank-login.com", "spam"),
        ("You have an unread message from Amazon Rewards. Claim your $250 voucher before it expires tonight.", "spam"),
        ("Get rich fast! Make $5,000 a day working from home with zero experience. Register now!", "spam"),
        ("Special promotion! Low interest rates on personal loans starting from 1.99%. Apply online today.", "spam"),
        ("Your bitcoin wallet has received 0.45 BTC. Click here to transfer to your bank account.", "spam"),
        ("Limited time offer: 90% discount on designer watches and luxury items. Shop now!", "spam"),
        ("Final warning: Your subscription will be cancelled unless you update your payment information today.", "spam"),

        # Ham (Legitimate) samples
        ("Hey, are we still meeting for lunch tomorrow at 1 PM?", "ham"),
        ("Please find attached the quarterly financial report for your review.", "ham"),
        ("Hi team, just a quick reminder about our project sync scheduled for 3 PM today.", "ham"),
        ("Can you send me the updated slides for the presentation on Monday?", "ham"),
        ("Thanks for your help with the code review earlier, appreciate it!", "ham"),
        ("Your doctor's appointment is confirmed for Thursday at 10:30 AM.", "ham"),
        ("Hey mom, I will be home around 6 PM. Do we need anything from the grocery store?", "ham"),
        ("Here is the meeting agenda and Zoom link for tomorrow's client call.", "ham"),
        ("Your order #48201 has shipped and is estimated to arrive on Friday.", "ham"),
        ("Could you please review and sign the attached document when you get a chance?", "ham"),
        ("Great job on the project release yesterday! Everything looks solid.", "ham"),
        ("Don't forget to submit your weekly timesheet before the end of the day.", "ham"),
        ("Hi John, let me know when you are free to discuss the budget allocation.", "ham"),
        ("The server maintenance is completed and all systems are running normally.", "ham")
    ]

    df = pd.DataFrame(data, columns=["text", "label"])

    # Vectorizer and Classifier
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    model = MultinomialNB()
    model.fit(X, y)

    # Save artifacts
    joblib.dump(model, "spam_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    print("SUCCESS: Model (spam_model.pkl) and Vectorizer (tfidf_vectorizer.pkl) trained and saved successfully!")

if __name__ == "__main__":
    train_and_save_model()
