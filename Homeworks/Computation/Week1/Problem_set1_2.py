# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:55:29 2017

@author: suket
"""
import numpy as np

'''Problem 1'''
def min_max_average(n):
    minimum, maximum, average = min(n), max(n), float(sum(n))/float(len(n))
    return minimum, maximum, average

A = [1, 3, 6, 11]


Answer1 = min_max_average(A)

print('\nProblem 1 Answer\n')
print(Answer1)

'''Problem 2'''
print('\n\nProblem 2 Answer\n')

#dictionaries
dict_1 = {1: 'x', 2: 'b'} # Create a dictionary.
dict_2 = dict_1 # Assign it a new name.
dict_2[1] = 'a' # Change the 'new' dictionary.
print('dictionaries')
if dict_1 == dict_2:
    print('--> mutable')
else:
    print('--> immutable')

#numbers
num_1 = 1
num_2 = num_1 # Assign it a new name.
num_2 += 1 
num_1 == num_2
print('numbers')
if num_1 == num_2:
    print('--> mutable')
else:
    print('--> immutable')

#strings
str_1 = 'a'
str_2 = str_1 # Assign it a new name.
str_2 += 'a' 
print('strings')
if str_1 == str_2:
    print('--> mutable')
else:
    print('--> immutable')

#lists
list_1 = [0]
list_2 = list_1 # Assign it a new name.
list_2.append(1)
print('lists') 
if list_1 == list_2:
    print('--> mutable')
else:
    print('--> immutable')

#tuples
tuple_1 = [0]
tuple_2 = tuple_1 # Assign it a new name.
tuple_2 += (1,)
print('tuples') 
if tuple_1 == tuple_2:
    print('--> mutable')
else:
    print('--> immutable')


'''Problem 3'''
import calculator as cal

def get_hypo(a, b):
    result = cal.sqrt(cal.summation(cal.product(a, a), cal.product(b, b)))
    return result

a = 3
b = 4
Answer3 = get_hypo(a, b)

print('\nProblem 3 Answer\n')
print(Answer3)
















