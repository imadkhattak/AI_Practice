class Student:

    def __init__(self, num1, num2):

        self.num1 = num1
        self.num2 = num2

    def addition(self):

        print("The sum of the 2 values is", self.num1 + self.num2)


student1 = Student(50, 70)
print(student1.num1)
print(student1.num2)
print(student1.addition())