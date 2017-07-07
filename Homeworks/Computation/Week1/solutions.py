# solutions.py
"""Volume IB: Testing.
<Name>
<Date>
"""
import math
import numpy as np
import random

# Problem 1 Write unit tests for addition().
# Be sure to install pytest-cov in order to see your code coverage change.


def addition(a, b):
    return a + b


def smallest_factor(n):
    """Finds the smallest prime factor of a number.
    Assume n is a positive integer.
    """
    if n == 1:
        return 1
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return i
    return n


# Problem 2 Write unit tests for operator().
def operator(a, b, oper):
    if type(oper) != str:
        raise ValueError("Oper should be a string")
    if len(oper) != 1:
        raise ValueError("Oper should be one character")
    if oper == "+":
        return a + b
    if oper == "/":
        if b == 0:
            raise ValueError("You can't divide by zero!")
        return a/float(b)
    if oper == "-":
        return a-b
    if oper == "*":
        return a*b
    else:
        raise ValueError("Oper can only be: '+', '/', '-', or '*'")

# Problem 3 Write unit test for this class.
class ComplexNumber(object):
    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)

    def norm(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __add__(self, other): #
        real = self.real + other.real
        imag = self.imag + other.imag
        return ComplexNumber(real, imag)

    def __sub__(self, other): #
        real = self.real - other.real
        imag = self.imag - other.imag
        return ComplexNumber(real, imag)

    def __mul__(self, other): #
        real = self.real*other.real - self.imag*other.imag
        imag = self.imag*other.real + other.imag*self.real
        return ComplexNumber(real, imag)

    def __truediv__(self, other):
        if other.real == 0 and other.imag == 0:
            raise ValueError("Cannot divide by zero")
        bottom = (other.conjugate()*other*1.).real
        top = self*other.conjugate()
        return ComplexNumber(top.real / bottom, top.imag / bottom)

    def __eq__(self, other):
        return self.imag == other.imag and self.real == other.real

    def __str__(self):
        return "{}{}{}i".format(self.real, '+' if self.imag >= 0 else '-',
                                                                abs(self.imag))

# Problem 5: Write code for the Set game here

def input_file(filename):
    if isinstance(filename, str) == True:
        with open(filename,'r') as in_file:
            contents = in_file.read()   
        in_file.closed
        return contents
    
    

def reg_value(filename):
    contents = input_file(filename)
    column_length = len(contents.split("\n")[0]) # It has to be "4".
    array = [''.join(i) for i in zip(contents.split())]

    #Card property check, is it 4-digits and also 0, 1, 2, 3?
    if len(contents) != 59:
        raise ValueError("The Property or the number of Cards are wrong...")
        

    
    #middle process for return right numbers
    join_array = array[0]
    for i in range(11):
        join_array += array[i+1]
    
        int_array = np.zeros(48)
        
    #More way to check the card is right or not.
    for i in range(12):
        if len(array[i]) != 4:
            raise ValueError("The Property of Card is wrong. (2)")
    
    
    #Are there duplicate cards here?
    for i in range(12):
        for j in range (i+1,12):
            for k in range (j+1,12):
                if array[k] == array[j] or array[k] == array[i]:
                    print("There are duplicate cards here.")
                    return False
                k = k+1
            j = j+1
        i = i+1
    
    # Final Process For return numbers from this function
    for i in range(48):
        int_array[i] = int(join_array[i])
      
    # Final Check! Property is 0 or 1 or 2 ???
    for i in range(len(int_array)):
        if int_array[i] == 0 or int_array[i] == 1 or int_array[i] == 2:
            #print("33")
            pass
        else:
            raise ValueError("The numbers of Card is definetely wrong. (not 0, 1, 2)")
    return int_array


# It nicely captures the set of the cards.
def set_the_card(int_array, step1 = 0, step2 = 0, step3 = 0):
    #step1 = input("pick the first card(from 1 to 12)  :")
    step_1 = int(step1)
    #step2 = input("pick the second card (it should be different from previous one)  :")
    step_2 = int(step2)
    #step3 = input("pick the first card (it should be different from previous one  :")
    step_3 = int(step3)
    
    if (int_array[step_1*4 - 4] + int_array[step_2*4 - 4] + int_array[step_3*4 - 4])%3 == 0:
        first = True
    else: first = False
    if (int_array[step_1*4 - 3] + int_array[step_2*4 - 3] + int_array[step_3*4 - 3])%3 == 0:
        second = True
    else: second = False
    if (int_array[step_1*4 - 2] + int_array[step_2*4 - 2] + int_array[step_3*4 - 2])%3 == 0:
        third = True
    else: third = False
    if (int_array[step_1*4 - 1] + int_array[step_2*4 - 1] + int_array[step_3*4 - 1])%3 == 0:
        forth = True
    else: forth = False

    if first == True and second == True and third == True and forth == True:
        return print("you picked right cards!")
    else:
        return print("you picked wrong cards!")





'''
#For experiment for this codes.

#reg_value("good.txt") 
#reg_value("bad_duplicate.txt")
#reg_value("bad_wrong_card2.txt")
#bad = reg_value("bad_wrong_card3.txt")
#bad_input = input_file("bad_wrong_card3.txt")
#input_file("filenameerror")
#aa = input_file("good.txt")
#reg_
'''

# NoW! Check the cards!!!!   

check = reg_value("good.txt") 
set_the_card(check, 1, 2, 3)  # picked right cards
set_the_card(check, 1, 2, 5)  # picked wrong cards

'''
We can change the codes above to the one to use "input" function in order to make user-friendly environment.
'''