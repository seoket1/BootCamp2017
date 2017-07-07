import numpy as np

'''Problem 1'''
import math

def arithmagic():
    step_1 = input("Enter a 3-digit number where the first and last "
                       "digits differ by 2 or more: ")
    if 100 <= int(step_1) <= 999:
        pass
    else:
        raise ValueError("'The first number (step_1) is not a 3-digit number.")

    if abs( int(step_1[0]) - int(step_1[2]) ) < 2:
        raise ValueError("The first number’s first and last digits differ by less than 2.")
    
    step_2 = input("Enter the reverse of the first number, obtained by reading it backwards: ")
    
    if step_1[0] == step_2[2] and step_1[2] == step_2[0]:
        pass
    else:
        raise ValueError("The second number (step_2) is not the reverse of the first number.")        
    
    step_3 = input("Enter the positive difference of these numbers: ")
    
    if abs( int(step_1) - int(step_2) ) == int(step_3):
        pass
    else:
        raise ValueError("The third number (step_3) is not the positive difference of the first two numbers.")    
    
    step_4 = input("Enter the reverse of the previous result: ")
    
    if step_3[0] == step_4[2] and step_3[2] == step_4[0]:
        final_result = int(step_3) + int(step_4)
    else:
        raise ValueError("The fourth number (step_4) is not the reverse of the third number.")
    
    print( str(step_3) + " + " + str(step_4) + " =  + {}".format(final_result), "(ta-da!)")
    
    
arithmagic()


'''Problem 2'''
from random import choice

def random_walk(max_iters=1e12):
    walk = 0
    direction = [1, -1]
    i=0
    for i in range(int(max_iters)):
        try:
            walk += choice(direction)
        except KeyboardInterrupt as e:
            break
        
    if max_iters == i + 1:
        print("Process completed")
        print("interation number :", i+1)
    else:
        print("Process interrupted at iteration", i)
        
    return walk


print("random walk = ", random_walk())



'''Problem 3'''
'''Problem 4'''

# To make txt file first.
with open("out_numbers.txt", 'w') as outfile: # Open 'out.txt' for writing.
    for i in range(10):
        outfile.write(str(i**2)+' ') # Write some strings (and spaces).
outfile.closed

with open("out_alphbets.txt", 'w') as outfile: # Open 'out.txt' for writing.
    for i in ["a", "B", "c", "D", "e", "F", "g"]:
        outfile.write(i) # Write some strings (and spaces).
    outfile.write("\n")
    for i in ["t", "y", "Y", "q", "g", "K", "z"]:
        outfile.write(i) # Write some strings (and spaces).
outfile.closed

with open("out.txt", 'w') as outfile: # Open 'out.txt' for writing.
    for i in range(10):
        outfile.write(str(i**2)+' ') # Write some strings (and spaces).
outfile.closed


class ContentFilter(object):
    def __init__(self, filename):
        try:
            if isinstance(filename, str) == 0:
                raise TypeError
        except TypeError as e:
            print("It should be a string.")
        else:
            self.filename = filename
            with open(filename,'r') as in_file:
                self.contents = in_file.read()   
            in_file.closed

    def uniform(self, copy_uniform, mode = "w", case = "upper"):
        if mode != "w" and mode != "a":
            raise ValueError("It should be w or r")
        if case != "upper" and case != "lower":
            raise ValueError("It should be upper or lower")
        if case == "upper":
            with open(copy_uniform, mode) as out_file_uniform:
                out_file_uniform.write(self.contents.upper())
            out_file_uniform.closed
        if case == "lower":
            with open(copy_uniform, mode) as out_file_uniform:
                out_file_uniform.write(self.contents.lower()) 
            out_file_uniform.closed
                
    def reverse(self, copy_reverse, mode = "w", unit = "line"):
        if mode != "w" and mode != "a":
            raise ValueError("It should be w or r")
        if unit != "word" and unit != "line":
            raise ValueError("It should be word or line.")
        if unit == "word":
            with open(copy_reverse, mode) as out_file_reverse:
                reverse_whole = "".join(reversed(self.contents))
                reverse = "\n".join(reversed(reverse_whole.split("\n")))
                out_file_reverse.write(reverse)
            out_file_reverse.closed
        if unit == "line":
            with open(copy_reverse, mode) as out_file_reverse:
                reverse = "\n".join(reversed(self.contents.split("\n")))
                out_file_reverse.write(reverse)
            out_file_reverse.closed
    
    def transpose(self, copy_transpose, mode = "w"):
        if mode != "w" and mode != "a":
            raise ValueError("It should be w or r")
       
        with open(copy_transpose, mode) as out_file_transpose:
            #This code below is for converting from string to array
            #transposed = [''.join(i) for i in zip(*self.contents.split())]
            
            #This code below is for string.
            transposed_string = '\n'.join(''.join(i) for i in zip(*self.contents.split()))
            out_file_transpose.write(transposed_string)
        out_file_transpose.closed



Wrong = ContentFilter(1)
out_numbers = ContentFilter("out_numbers.txt")
print("\nFile name is", out_numbers.filename)
print("File contents are", out_numbers.contents)

out_alphabets = ContentFilter("out_alphbets.txt")
print("\nFile name is", out_alphabets.filename)
print("File contents are", out_alphabets.contents)

out_alphabets.uniform("copy_uniform.txt")
out_alphabets.uniform("copy_uniform.txt", "a", "lower")
out_alphabets.uniform("copy_uniform.txt", "w", "upper")
out_alphabets.uniform("copy_uniform.txt", "a", "lower")
#out_alphabets.uniform("copy_uniform.txt", "a", "45566")

out_alphabets.reverse("copy_reverse_word.txt", "w", "word")
out_alphabets.reverse("copy_reverse_line.txt", "w", "line")

out_alphabets.transpose("copy_transpose.txt", "w")
