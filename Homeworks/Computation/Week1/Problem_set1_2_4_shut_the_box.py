# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 21:21:18 2017

@author: suket
"""

'''Problem 4'''
import math
import sys
import random
import box as box

if len(sys.argv) == 1:
    print("At least one extra command line argument is required")
    Name = input("Please write your name :")
    if len(Name) == 0:
        print("The program ends. Please restart.")
        exit()
    else:
        pass
else:    
    print(sys.argv)
    
    #by sys.argv, make natural nickname.
    Name = sys.argv[1]
    for i in range(len(sys.argv)):
        if i+2 >= len(sys.argv):
            print
            break
        else:
            Name += " "+sys.argv[i+2]



print("Welcome,", Name, "!")



Numbers = [1,2,3,4,5,6,7,8,9] 

while len(Numbers) > 0:
    dice1 = range(1, 7)
    dice2 = range(1, 7)
    sum_of_dices = random.choice(dice1) + random.choice(dice2) 

    print("Numbers left: ", Numbers)
    print("Roll: ", sum_of_dices)
    
    
    if box.isvalid(sum_of_dices, Numbers) == 1:
        pass
    else:
        score = sum(Numbers)
        print("Game over!\n")
        print("Score for player ", Name,": ", score)
        sys.exit(1)
        
    
    chosen_numbers = input("Numbers to eliminate: ")
    choices = box.parse_input(chosen_numbers, Numbers)
    
    if box.isvalid(sum_of_dices, choices) == 0:
        chosen_numbers = input("Please Provide Right Numbers to eliminate: ")  
        choices = box.parse_input(chosen_numbers, Numbers)
        if box.isvalid(sum_of_dices, choices) == 0:
            print("The program ends. Please restart after learning the game.")
            sys.exit(1)
        elif choices == []:
            print("The program ends. Please restart after learning the game.")
            sys.exit(1)   
        else:
            pass
    elif choices == []:
        chosen_numbers = input("Please Provide Right Numbers to eliminate: ")
        choices = box.parse_input(chosen_numbers, Numbers)
        if box.isvalid(sum_of_dices, choices) == 0:
            print("The program ends. Please restart after learning the game.")
            sys.exit(1)
        elif choices == []:
            print("The program ends. Please restart after learning the game.")
            sys.exit(1) 
        else:
            pass
    else:
        pass
    
    
    choices = box.parse_input(chosen_numbers, Numbers)
    #print(chosen_numbers)

    ###########################
    print(choices)

    for i in range(len(Numbers)):
        #print(i)
        for j in range(len(choices)):
            #print(j)
            if i >= len(Numbers):
                break
            else:
                if Numbers[i] == choices[j]:
                    Numbers.remove(Numbers[i])   #why 7 exists here?
                    #print(Numbers)
                    #print('if')
                else:
                   # print(Numbers)
                    #print('else')
                    pass

print("Score for player ", Name, ": 0 points")
print("Congratulations!! you shut the box!")

