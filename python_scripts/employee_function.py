class Employee:
    # Class Attribute (Shared by all employees)
    company_name = "Apex AI Labs"
    total_employees = 0

    def __init__(self, name, role):
        # Instance Attributes (Unique per employee)
        self.name = name
        self.role = role
        Employee.total_employees += 1

emp1 = Employee("Alice", "AI Engineer")
emp2 = Employee("Bob", "Data Scientist")

emp3 = Employee("Ali", "DevOPs")

print(emp1.company_name)  # Apex AI Labs
print(emp2.company_name)  # Apex AI Labs
print(emp3.company_name)

print(Employee.total_employees) # 2

# Modifying class attribute updates everywhere
Employee.company_name = "Global Apex AI"
print(emp1.company_name)  # Global Apex AI
