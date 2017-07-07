# test_solutions.py
"""Volume 1B: Testing.
<Name>
<Class>
<Date>
"""

import solutions as soln
import pytest
import math
from solutions import addition

''' Problem 1: Test the addition and fibonacci functions from solutions.py
----------- coverage: platform win32, python 3.6.1-final-0 -----------
Name                Stmts   Miss  Cover
---------------------------------------
solutions.py           57     33    42%
test_solutions.py      26      0   100%
---------------------------------------
TOTAL                  83     33    60%


========================== 5 passed in 0.13 seconds ===========================
'''


def test_addition():
    assert soln.addition(1,3) == 4, "Addition failed on positive integers"
    assert soln.addition(-5,-7) == -12, "Addition failed on negative integers"
    assert soln.addition(-6,14) == 8, "Addition failed on negative integer and positive integers"

def test_smallest_factor():
    assert soln.smallest_factor(1) == 1, "Finding smallest factor failed on 1"
    assert soln.smallest_factor(5) == 5, "Finding smallest factor failed on 5"
    assert soln.smallest_factor(8) == 2, "Finding smallest factor failed on 8"
'''
----------- coverage: platform win32, python 3.6.1-final-0 -----------
Name                Stmts   Miss  Cover
---------------------------------------
solutions.py           57     26    54%
test_solutions.py      32      0   100%
---------------------------------------
TOTAL                  89     26    71%


========================== 5 passed in 0.08 seconds ===========================
'''

'''Problem 2: Test the operator function from solutions.py'''
def test_operator():
    with pytest.raises(Exception) as excinfo:
        soln.operator(1, 1, 1)
    assert excinfo.typename == "ValueError"
    assert excinfo.value.args[0] == "Oper should be a string"
    
    with pytest.raises(Exception) as excinfo:
        soln.operator(1, 1, "Test 1")
    assert excinfo.typename == "ValueError"
    assert excinfo.value.args[0] == "Oper should be one character"
    
    assert soln.operator(3, 5, "+") == 8, "Addition failed on positive numbers"
    
    assert soln.operator(10, 5, "/") == 2, "Division failed on positive numbers"
    
    assert soln.operator(5, 3, "-") == 2, "Subtraction failed on positive numbers"
    
    assert soln.operator(3, 5, "*") == 15, "Multipying failed on positive numbers"   
    
    with pytest.raises(Exception) as excinfo:
        soln.operator(4, 0, "/")
    assert excinfo.typename == "ValueError"
    assert excinfo.value.args[0] == "You can't divide by zero!"

    with pytest.raises(Exception) as excinfo:
        soln.operator(4, 5, "a")
    assert excinfo.typename == "ValueError"
    assert excinfo.value.args[0] == "Oper can only be: '+', '/', '-', or '*'"
    
'''
----------- coverage: platform win32, python 3.6.1-final-0 -----------
Name                Stmts   Miss  Cover
---------------------------------------
solutions.py           57     11    81%
test_solutions.py      51      0   100%
---------------------------------------
TOTAL                 108     11    90%


========================== 5 passed in 0.11 seconds ===========================
'''  

'''Problem 3: Finish testing the complex number class'''
@pytest.fixture
def set_up_complex_nums():
    number_1 = soln.ComplexNumber(1, 2)
    number_2 = soln.ComplexNumber(5, 5)
    number_3 = soln.ComplexNumber(2, 9)
    return number_1, number_2, number_3

def test_complex_conjugate(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1.conjugate() == soln.ComplexNumber(1, -2)
    assert number_2.conjugate() == soln.ComplexNumber(5, -5)
    assert number_3.conjugate() == soln.ComplexNumber(2, -9)

def test_complex_norm(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1.norm() == math.sqrt(5)
    assert number_2.norm() == math.sqrt(50)
    assert number_3.norm() == math.sqrt(85)

def test_complex_addition(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 + number_2 == soln.ComplexNumber(6, 7)
    assert number_1 + number_3 == soln.ComplexNumber(3, 11)
    assert number_2 + number_3 == soln.ComplexNumber(7, 14)
    assert number_3 + number_3 == soln.ComplexNumber(4, 18)

def test_complex_subtraction(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 - number_2 == soln.ComplexNumber(-4, -3)
    assert number_1 - number_3 == soln.ComplexNumber(-1, -7)
    assert number_2 - number_3 == soln.ComplexNumber(3, -4)
    assert number_3 - number_3 == soln.ComplexNumber(0, 0)
    
def test_complex_multiplication(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert number_1 * number_2 == soln.ComplexNumber(-5, 15)
    assert number_1 * number_3 == soln.ComplexNumber(-16, 13)
    assert number_2 * number_3 == soln.ComplexNumber(-35, 55)
    assert number_3 * number_3 == soln.ComplexNumber(-77, 36)    
    
def test_complex_division(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums

    assert number_1/number_2 == soln.ComplexNumber(0.3,0.1)
    assert number_1/number_3 == soln.ComplexNumber(4/17,-1/17)
    assert number_2/number_3 == soln.ComplexNumber(11/17,-7/17)
    assert number_3/number_3 == soln.ComplexNumber(1,0)
    
    number_00 = soln.ComplexNumber(0,0)
    with pytest.raises(Exception) as excinfo:
        number_2/number_00
    assert excinfo.typename == "ValueError"
    assert excinfo.value.args[0] == "Cannot divide by zero"

# Alreay I did a test about "equality" above.
#def __str__(self):
#    return "{}{}{}i".format(self.real, '+' if self.imag >= 0 else '-', abs(self.imag))


def test_complex_string(set_up_complex_nums):
    number_1, number_2, number_3 = set_up_complex_nums
    assert str(number_1) == "1+2i"
    assert str(number_2) == "5+5i"
    assert str(number_3) == "2+9i"
    
'''
----------- coverage: platform win32, python 3.6.1-final-0 -----------
Name                Stmts   Miss  Cover
---------------------------------------
solutions.py           60      0   100%
test_solutions.py      84      0   100%
---------------------------------------
TOTAL                 144      0   100%
'''    
    


'''Problem 4: Write test cases for the Set game.'''

def test_set_the_card():

    assert soln.input_file("good.txt"), "file name error."

    assert len(soln.reg_value("good.txt")) == 48, "Card number fails"
    assert soln.reg_value("bad_duplicate.txt") == False, "You cannot capture dupicate cards!"
    
    with pytest.raises(Exception) as excinfo:
        soln.reg_value("bad_not12.txt")
    assert excinfo.typename == "ValueError"
    assert excinfo.value.args[0] == "The Property or the number of Cards are wrong..."
    
    
    with pytest.raises(Exception) as excinfo:
        soln.reg_value("bad_wrong_card2.txt")
    assert excinfo.typename == "ValueError"
    assert excinfo.value.args[0] == "The Property of Card is wrong. (2)"
    
    with pytest.raises(Exception) as excinfo:
        soln.reg_value("bad_wrong_card3.txt")
    assert excinfo.typename == "ValueError"
    assert excinfo.value.args[0] == "The numbers of Card is definetely wrong. (not 0, 1, 2)"

    assert soln.set_the_card(soln.reg_value("good.txt"), 1, 2, 3) == print("you picked right cards!"), "Something wrong."
    assert soln.set_the_card(soln.reg_value("good.txt"), 1, 2, 5) == print("you picked wrong cards!"), "Something wrong."
    assert soln.set_the_card(soln.reg_value("good.txt"), 1, 2, 6) == print("you picked wrong cards!"), "Something wrong."

'''

============================= test session starts =============================
test_solutions.py ...........

----------- coverage: platform win32, python 3.6.1-final-0 -----------
Name                Stmts   Miss  Cover
---------------------------------------
solutions.py          117      0   100%
test_solutions.py     103      0   100%
---------------------------------------
TOTAL                 220      0   100%


========================== 11 passed in 0.67 seconds ==========================
'''








'''
 For Pratice.
 
 
 
    with pytest.raises(Exception) as excinfo:
        soln.reg_value("good.txt")
    assert excinfo.typename == "ValueError"
    assert excinfo.value.args[0] == "The Property or the number of Cards are wrong."
    
    assert len(soln.reg_value("good.txt")) == 48, "Card number fails"
    
    with pytest.raises(Exception) as excinfo:
        soln.set_the_card(soln.reg_value("good.txt"))
    assert excinfo.typename == "ValueError"
    assert excinfo.value.args[0] == "The Property or the number of Cards are wrong."

    with pytest.raises(Exception) as excinfo:
        soln.reg_value("bad_duplicate.txt")
    assert excinfo.typename == "ValueError"
    assert excinfo.value.args[0] == "There are duplicate cards."
    
    with pytest.raises(Exception) as excinfo:
        soln.reg_value("bad_not12.txt")
    assert excinfo.typename == "ValueError"
    assert excinfo.value.args[0] == "The Property or the number of Cards are wrong..."
    
'''

