import csv
import sys
from shopping import load_data

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    print("start")
    evidence, labels = load_data("shopping.csv")
    # print(labels)
    # print(evidence[0])
    
    
main()