import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4
MONTH_MAP = {
    "Jan": 0,
    "Feb": 1,
    "Mar": 2,
    "Apr": 3,
    "May": 4,
    "June": 5,
    "Jul": 6,
    "Aug": 7,
    "Sep": 8,
    "Oct": 9,
    "Nov": 10,
    "Dec": 11
}


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as fp:
        data = [[], []]
        fp.readline()   # skip the header line
        next_line = load_a_line(fp)
        while next_line:
            data[0].append(next_line[0])
            data[1].append(next_line[1])
            next_line = load_a_line(fp)

    return data    

def load_a_line(fp):
    """
    Load a line from the load shopping data.
    """
    data = fp.readline()
    if not data:    # end of the file
        return None
    data = data.split(",")
    evidence = data[: 17]
    label = data[17][:-1]    # escape the new line chacter
    for i in range(17):
        if i in (0, 2, 4, 11, 12, 13, 14):
            evidence[i] = int(evidence[i])
        if i in (1, 3, 5, 6, 7, 8, 9):
            evidence[i] = float(evidence[i])
    evidence[10] = MONTH_MAP[evidence[10]]
    if evidence[15] == "Returning_Visitor":
        evidence[15] = 1
    else:
        evidence[15] = 0
    if evidence[16] == "True":
        evidence[16] = 1
    else:
        evidence[16] = 0
    
    if label == "TRUE":
        label = 1
    else:
        label = 0

    return [evidence, label]
        


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    p = 0
    tp = 0
    f = 0
    tf = 0
    for y, predict in zip(labels, predictions):
        if y == 1:
            p += 1
            if y == predict:
                tp += 1
        else:
            f += 1
            if y == predict:
               tf += 1

    return (tp / p, tf / f)

if __name__ == "__main__":
    main()
