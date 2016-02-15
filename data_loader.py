import csv
import numpy as np

def load(filepath, exclude=[3], standardize=True):

    num_lines = sum(1 for line in open(filepath))
    with open(filepath) as f:

        reader = csv.reader(f)
        X = None
        Y = []
        i = 0
        for row in reader:
            sample = get_features(row[:-1], exclude)
            if X is None: X = np.empty((num_lines, len(sample)), dtype="float32")
            X[i,:] = sample
            Y.append(1 if row[-1] == " True." else 0)
            i += 1

        # Optionally standardize non-binary dimensions
        if standardize:
            stds = np.std(X, axis=0)
            X[:,51:] /= stds[51:]

        return (X, Y)

def get_features(data_row, exclude=[]):
    # State (0)
    states = ['KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT', 'NY', 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'NE', 'WY', 'HI', 'IL', 'NH', 'GA', 'AK', 'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA', 'NM', 'NV', 'DC', 'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND']
    features = np.zeros(51)
    features[states.index(data_row[0])] = 1

    # International plan (4)
    features = np.append(features, 1.0 if data_row[4] == " yes" else 0.0)

    # Voice mail plan (5)
    features = np.append(features, 1.0 if data_row[5] == " yes" else 0.0)

    # Add the rest
    idxs = range(20)
    exclude += [0,4,5]
    exclude = set(sorted(exclude))
    for e in exclude:
        idxs.remove(e)
    for i in idxs:
        features = np.append(features, float(data_row[i]))

    return features

def get_feature_names(exclude=[3]):
    names = ['KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI', 'IA', 'MT', 'NY', 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC', 'NE', 'WY', 'HI', 'IL', 'NH', 'GA', 'AK', 'MD', 'AR', 'WI', 'OR', 'MI', 'DE', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA', 'NM', 'NV', 'DC', 'KY', 'ME', 'MS', 'TN', 'PA', 'CT', 'ND']
    names += ["Internatioal_plan", "Voicemail_plan"]
    fields = [
        "state",
        "account length",
        "area code",
        "phone number",
        "international plan",
        "voice mail plan",
        "number vmail messages",
        "total day minutes",
        "total day calls",
        "total day charge",
        "total eve minutes",
        "total eve calls",
        "total eve charge",
        "total night minutes",
        "total night calls",
        "total night charge",
        "total intl minutes",
        "total intl calls",
        "total intl charge",
        "number customer service calls"]
    idxs = range(20)
    exclude += [0,4,5]
    exclude = set(sorted(exclude))
    for e in exclude:
        idxs.remove(e)
    for i in idxs:
        names.append(fields[i])

    return names
    
if __name__ == "__main__":
    (X, Y) = load("Dataset/churn.data.txt")
    print X
