from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

training_data = [
    # Bar chart (category-value pair)
    ([
        {"category": "A", "value": 10},
        {"category": "B", "value": 20}
    ], "bar"),

    # Line chart (time series)
    ([
        {"date": "2023-01-01", "value": 5},
        {"date": "2023-01-02", "value": 15}
    ], "line"),

    # Pie chart (part of whole)
    ([
        {"label": "Apple", "value": 40},
        {"label": "Banana", "value": 60}
    ], "pie"),

    # Scatter plot (x-y numerical)
    ([
        {"x": 1, "y": 2},
        {"x": 3, "y": 6}
    ], "scatter"),
]


def extract_features(data):
    first = data[0]
    keys = list(first.keys())
    types = [type(first[k]).__name__ for k in keys]
    features = {
        "num_fields": len(keys),
        "num_str": sum(1 for t in types if t == 'str'),
        "num_int": sum(1 for t in types if t == 'int'),
        "has_date": any('date' in k.lower() for k in keys),
        'has_category': any(k.lower() in ['category', 'label'] for k in keys),
        'has_xy': all(k in keys for k in ['x', 'y'])
    }
    return list(features.values())


x = [extract_features(d) for d, label in training_data]
y = [label for d, label in training_data]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier()
model.fit(x, y_encoded)


def predict_chart_type(new_data):
    features = extract_features(new_data)
    pred = model.predict([features])
    return le.inverse_transform(pred)[0]


test_input = [
    {"category": "X", "value": 100},
    {"category": "Y", "value": 200}
]


print("Suggested chart type:", predict_chart_type(test_input))
