import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = {
    'text': [
        'I love this product, it is amazing!',
        'Horrible experience, not recommended.',
        'It is okay, nothing special.',
        'Absolutely fantastic, exceeded my expectations.',
        'Worst purchase ever, very disappointed.',
        'Neutral feelings about this item.',
        'Great value and quality for the price.',
        'Not bad, but could be better.',
        'Terrible customer service!',
        'Satisfied with the product overall.',
        
        'This is the best thing I have bought this year!',
        'Completely useless and a waste of money.',
        'Average performance, nothing impressive.',
        'Superb quality, highly recommend it.',
        'Extremely poor build quality.',
        'Does the job, nothing more nothing less.',
        'Really happy with this purchase.',
        'Could be improved in some areas.',
        'Very disappointing experience overall.',
        'I am quite satisfied with it.',
        
        'Excellent product, works flawlessly.',
        'Bad packaging and damaged item received.',
        'It is decent for the price.',
        'Loved it, will buy again!',
        'Not worth the price at all.',
        'It is fine, not too good not too bad.',
        'Amazing performance and great design.',
        'Mediocre experience.',
        'Worst customer support ever.',
        'Pretty good overall.',
        
        'Highly satisfied with the results.',
        'Terrible quality, broke in a week.',
        'Nothing extraordinary about it.',
        'Fantastic product and fast delivery.',
        'Very poor experience.',
        'It works as expected.',
        'Great purchase, very happy!',
        'Slightly better than expected.',
        'Not happy with the product.',
        'Overall it is okay.',
        
        'Super happy with this!',
        'Regret buying this product.',
        'It is just average.',
        'Exceeded all my expectations!',
        'Very bad quality.',
        'Neither good nor bad.',
        'Great features and easy to use.',
        'Could have been much better.',
        'Awful experience.',
        'I am okay with this purchase.',
        
        'Brilliant product!',
        'Do not buy this, very bad.',
        'Nothing special, just normal.',
        'Loved the quality and performance.',
        'Worst experience ever.',
        'It is acceptable.',
        'Very good product for daily use.',
        'Somewhat disappointing.',
        'Extremely bad service.',
        'Fair enough for the price.'
    ],
    
    'sentiment': [
        'positive','negative','neutral','positive','negative',
        'neutral','positive','neutral','negative','positive',
        
        'positive','negative','neutral','positive','negative',
        'neutral','positive','neutral','negative','positive',
        
        'positive','negative','neutral','positive','negative',
        'neutral','positive','neutral','negative','positive',
        
        'positive','negative','neutral','positive','negative',
        'neutral','positive','neutral','negative','neutral',
        
        'positive','negative','neutral','positive','negative',
        'neutral','positive','neutral','negative','neutral',
        
        'positive','negative','neutral','positive','negative',
        'neutral','positive','neutral','negative','neutral'
    ]
}

df = pd.DataFrame(data)


X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
