# Gender Prediction – Exxeta Challenge

## Daten

Format: user_id, path, timestamp, gender

- **path** = besuchte Seite  
- **gender** = Zielvariable (nur im Training)

---

## Vorgehen

###  Aggregation pro User
Alle besuchten Seiten werden pro User zu einem Dokument zusammengefasst:
python: 
```
train_docs = train_data.groupby("user_id")["path"].apply(lambda x: " ".join(x))
test_docs  = test_data.groupby("user_id")["path"].apply(lambda x: " ".join(x))
```

### Seltene Seiten entfernen
<5 Vorkommen werden entfernt.
python:
``
vec = CountVectorizer(mind_df=5)
``

## Feature Extraction

Paths werden mit CountVectorizer in numerische Features umgewandelt:

vec = CountVectorizer(min_df=5)
X      = vec.fit_transform(train_docs)
X_test = vec.transform(test_docs)

## Zusatzfeatures

Zusätzlich werden User-Verhaltensfeatures genutzt:

total_visits (Klickanzahl)

unique_paths (verschiedene Seiten)

num_sessions (Pause > 30min)

Diese werden an X und X_test angehängt.

## Modell

Klassifikation mit Logistic Regression:

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

## Output

Vorhersagen für test.csv werden gespeichert als:

./data/predictions.csv

## Requirements
pip install pandas numpy scikit-learn scipy