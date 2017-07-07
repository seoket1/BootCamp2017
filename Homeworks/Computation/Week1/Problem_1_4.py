import math


'''Problem 1'''
'''Problem 3'''

class Backpack(object):
    def __init__(self, name, color, max_size = 5): # This function is the constructor.
        self.name = name # Initialize some attributes.
        self.color = color
        self.max_size = max_size
        self.contents = []
    
    def put(self, item):
        """Add 'item' to the backpack's list of contents."""
        if len(self.contents) > (self.max_size - 1):
            print("\n ***Wanrning! No Room!***")
        else:
            self.contents.append(item)
    
    def take(self, item):
        """Remove 'item' from the backpack's list of contents."""
        self.contents.remove(item)

    def dump(self):
        self.contents = []
        
    def __eq__(self, other):
        if self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents):
            return True
        else:
            return False
    
    def __ne__(self, other):
        return not self == other

    def __str__(self):
        Owner = "\nOwner:  \t {}".format(self.name) #+ self.name
        Color = "\nColor:  \t {}".format(self.color)
        Size = "\nSize:  \t\t {}".format(len(self.contents))
        Max_size = "\nMax Size:  \t {}".format(self.max_size)
        Contents = "\nContents:  \t {}".format(self.contents)
        return Owner + Color + Size + Max_size + Contents
        
# For the test (This is working outside this code file.)
def test_backpack():
    print ("///////  backpack  ///////")
    testpack = Backpack("Barry", "black") # Instantiate the object.
    if testpack.max_size != 5: # Test an attribute.
        print("Wrong default max_size!")
    for item in ["pencil", "pen", "paper", "computer", "notebook", "book"]:
        print("\n--adding contents ===> " + item)
        testpack.put(item) # Test a method.
        print(testpack)
    testpack.take("paper")
    print("\n--take paper out--")
    print(testpack)
    
    print("\n\n ********** For checking that equality option is working.")
    testpack_2 = Backpack("Barry", "black") # Instantiate the object.
    if testpack_2.max_size != 5: # Test an attribute.
        print("Wrong default max_size!")
    for item in ["pencil", "pen", "paper", "computer", "notebook", "book"]:
        print("\n--adding contents ===> " + item)
        testpack_2.put(item) # Test a method.
        print(testpack_2)
    testpack_2.take("paper")
    print("\n--take paper out--")
    print(testpack_2)
    
    print("\n\n Is it same??")
    print(testpack == testpack_2)
    
    '''
    testpack.dump()
    print("\n--dump--")        
    print(testpack)
    '''


print("\n\n\n")
test_backpack()
print("\n\n\n")




'''Problem 2'''

class Jetpack(Backpack):
    def __init__(self, name, color, max_size = 2, fuel=10):
        Backpack.__init__(self, name, color, max_size)
        self.fuel = fuel

    def fly(self,fuel):
        if self.fuel >= fuel:
            self.fuel = self.fuel - fuel
        else:
            print("\n***Not Enough Fuel!!!***")

    def dump(self):
        self.contents = []
        self.fuel = 0

# For the test (This is working outside this code file.)
def test_Jetpack():
    print ("///////  Jestpack  /////// I changed max_limit is 2 here.")
    testpack = Jetpack("Barry", "black") # Instantiate the object.
    if testpack.max_size != 2: # Test an attribute.
        print("Wrong default max_size!")
    for item in ["pencil", "pen", "paper", "computer", "notebook", "book"]:
        print("\n--adding contents ===> " + item)
        testpack.put(item) # Test a method.
        print(testpack.name)
        print(testpack.color)
        print(testpack.contents)
        print(testpack.fuel)
    
    print("\n--take pencil out--")    
    testpack.take("pencil")
    print(testpack.name)
    print(testpack.color)
    print(testpack.contents)
    print(testpack.fuel)

    print("\n--fly--     5")  
    testpack.fly(5)
    print(testpack.name)
    print(testpack.color)
    print(testpack.contents)
    print(testpack.fuel)
    
    print("\n--fly--     8")    
    testpack.fly(8)
    print(testpack.name)
    print(testpack.color)
    print(testpack.contents)
    print(testpack.fuel)
    testpack.dump()
    
    print("\n--dump--")        
    print(testpack.name)
    print(testpack.color)
    print(testpack.contents)
    print(testpack.fuel)



print("\n\n\n")
test_Jetpack()
print("\n\n\n")
        
        
        

'''Preblem 4'''

        
class ComplexNumber(object):
    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)
    
    def __abs__(self):
        return math.sqrt(self.real**2 + self.imag**2)
    
    def __lt__(self, other):
        return self.real**2 + self.imag**2 < other.real**2 + other.imag**2
    
    def __gt__(self, other):
        return self.real**2 + self.imag**2 > other.real**2 + other.imag**2
    
    def __eq__(self, other):
        return self.imag == other.imag and self.real == other.real
    
    def __ne__(self, other):
        return not self == other

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

    def __str__(self):
        return "{}{}{}i".format(self.real, '+' if self.imag >= 0 else '-',
                                                                abs(self.imag)) 
        
        

        
        
        
        
        
        
        
        