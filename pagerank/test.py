from pagerank import *

example1 = {"1.html": {"2.html", "3.html"}, 
            "2.html": {"3.html"}, "3.html": {"2.html"}}
ex1page = "1.html"

print(transition_model(example1, ex1page, 0.85))
