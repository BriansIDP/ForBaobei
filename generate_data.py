import random
import pprint

import numpy as np

def get_age():
    age = np.random.normal(loc=25, scale=15)
    if age <= 1:
        age = 1
    elif age >= 100:
        age = 100
    return int(age)

def get_gender():
    return random.choice(['male', 'female'])

def get_education(age):
    if age <= 6:
        return random.choice(['infant', 'junior'])
    elif age <= 15:
        return random.choice(['junior', 'secondary'])
    else:
        return random.choice(['junior', 'secondary', 'university'])

def get_occupation(jobs, age):
    if age <= 3:
        return 'infant'
    elif age <= 18:
        return random.choice(['student'])
    else:
        return random.choice(jobs)

def get_nationality(countries):
    return random.choice(countries)

def generate(num_of_users, occupation_list, country_list):
    user_list = []
    jobs = []
    countries = []
    with open(occupation_list) as fin:
        for line in fin:
            jobs.append(line.strip())

    with open(country_list) as fin:
        for line in fin:
            countries.append(line.strip().lower())

    for i in range(num_of_users):
        user = {}
        user['age'] = get_age()
        user['gender'] = get_gender()
        user['education'] = get_education(user['age'])
        user['nationality'] = get_nationality(countries)
        user['occupation'] = get_occupation(jobs, user['age'])
        user_list.append(user)
    return user_list

if __name__ == "__main__":
    users = generate(10, 'jobs.txt', 'countries.txt')
    for user in users:
        pprint.pprint(user)
