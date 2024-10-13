import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])
    
    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    calculated = set()
    jp = 1       # the joint probability
    factors = {}    # factor of mutiply contributed by each person
    while len(calculated) != len(people):
        predessors = find_predessors(people, calculated)
        calculated |= predessors

        for person in predessors:
            mother = people[person]['mother']
            father = people[person]['father']

            # case 1, no parent info
            if mother == father == None:
                if person in one_gene:
                    factors[person] = PROBS['gene'][1]
                elif person in two_genes:
                    factors[person] = PROBS["gene"][2]
                else: 
                    factors[person] = PROBS['gene'][0]

            # case 2, parent info is known
            else:
                if person in one_gene:
                    factors[person] = \
                        pass_one_prob(father, one_gene, two_genes) * (1 - pass_one_prob(mother, one_gene, two_genes)) + \
                        (1 - pass_one_prob(father, one_gene, two_genes)) * pass_one_prob(mother, one_gene, two_genes)
                elif person in two_genes:
                    # P(two genes|mother info, father info)
                    factors[person] = pass_one_prob(father, one_gene, two_genes) * \
                        pass_one_prob(mother, one_gene, two_genes)
                else:
                    factors[person] = (1 - pass_one_prob(father, one_gene, two_genes)) * \
                        (1 - pass_one_prob(mother, one_gene, two_genes))
    
    # multiply all factors together
    for factor in factors.values():
        jp *= factor
    
    # include the probility of having/not having trait
    for person in people:
        trait_status = person in have_trait
        if person in one_gene:
            gene_num = 1
        elif person in two_genes:
            gene_num = 2
        else:
            gene_num = 0
        jp *= PROBS["trait"][gene_num][trait_status]
    
    return jp

def pass_one_prob(person, one_gene, two_genes):
    """
    Probability that `person pass one gene to his/her child.
    """
    if person in one_gene:
        return 0.5 * (1 - PROBS['mutation']) + 0.5 * PROBS['mutation']
    elif person in two_genes:
        return 1 - PROBS['mutation']
    else:
        return PROBS['mutation']
            
    
def find_predessors(people, parents):
    """
    Return people either not have info of parents or parents is in
    set `parents.
    """
    predessors = set()
    for person, info in people.items():
        if person not in parents:
            if (info['mother'] == info['father'] == None) or \
                (info['mother'] in parents or info['father'] in parents):
                 predessors.add(person)

    return predessors


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person, record in probabilities.items():
        if person in one_gene:
            record['gene'][1] += p
        elif person in two_genes:
            record['gene'][2] += p
        else:
            record['gene'][0] += p
        
        record['trait'][person in have_trait] += p
            


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for record in probabilities.values():
        gene_c = 1 / sum(record['gene'].values())
        trait_c = 1 / sum(record['trait'].values())

        for gene in record['gene']:
            record['gene'][gene] *= gene_c
        
        for trait in record['trait']:
            record['trait'][trait] *= trait_c


if __name__ == "__main__":
    main()
