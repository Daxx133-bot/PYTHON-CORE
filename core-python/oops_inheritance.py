import random
class BankInfo:
    def __init__(self):
        self.firstname = input("Enter First Name: ")
        self.lastname = input("Enter Last Name: ")
        self.gender = input("Enter Gender: ")
        self.address = input("Enter Address: ")
class BankAccount:
    def __init__(self, info, amount):
        self.object_bank_info = info
        self.acno = random.randint(1000000000000, 9999999999999)
        self.amount = amount

class Saving(BankAccount):
    def __init__(self, info, amount):
        super().__init__(info, amount)
        self.min_amount = 10000
        self.rate = 6
    def calculate_interest(self, months):
        interest = (self.amount * self.rate * months) / (100 * 12)
        return interest

class Current(BankAccount):
    def __init__(self, info, amount):
        super().__init__(info, amount)
        self.min_amount = 5000
        self.rate = None

class Main:
    def display(self):
        info = BankInfo()
        choice = input("Enter Account Type(Saving/Current): ")
        attempts = 0
        while attempts < 3:

            amount = int(input("Enter Amount: "))

            if choice.lower() == "saving":

                if amount >= 10000:
                    acc = Saving(info, amount)
                    months = int(input("Enter Months: "))

                    interest = acc.calculate_interest(months)

                    print("First Name:", acc.object_bank_info.firstname)
                    print("Last Name:", acc.object_bank_info.lastname)
                    print("Gender:", acc.object_bank_info.gender)
                    print("Address:", acc.object_bank_info.address)
                    print("Account Number:", acc.acno)
                    print("Amount:", acc.amount)
                    print("Months:", months)
                    print("Rate of Interest:", acc.rate)
                    print("Interest:", interest)
                    break
                else:
                    print("Minimum Amount should be 10000")
            elif choice.lower() == "current":
                if amount >= 5000:

                    acc = Current(info, amount)
                    print("First Name:", acc.object_bank_info.firstname)
                    print("Last Name:", acc.object_bank_info.lastname)
                    print("Gender:", acc.object_bank_info.gender)
                    print("Address:", acc.object_bank_info.address)
                    print("Account Number:", acc.acno)
                    print("Amount:", acc.amount)
                    print("Minimum Balance:", acc.min_amount)

                    break
                else:
                    print("Minimum Amount should be 5000")
            else:
                print("Invalid Account Type")

                break
            attempts += 1
        if attempts == 3:
            print("you are out of attempts")
            print("Program Terminated")

obj = Main()
obj.display()
