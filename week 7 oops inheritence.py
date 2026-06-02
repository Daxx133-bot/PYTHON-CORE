import random
class Bankinfo:
    def __init__(self):
        self.firstname = input("Enter your first name: ")
        self.lastname = input("Enter your last name: ")
        self.gender = input("enter your gender:")
        self.address = input("enter your address:")


class BankAccount(Bankinfo):
    def __init__(self,amount):
        self.amount = int(input("enter your amount:"))
        self.acno = random.randint(1000000000000, 9999999999999)

class Saving(BankAccount):
    def __init__(self,min_amount,rate):
        super().__init__(min_amount,rate):
        self.min_amount = 1000
        self.rate = 6
class Current(BankAccount):
    def __init__(self,min_amount,rate):
        self.min_amount = 5000
        self.rate = None

class Main:
    def display(self):
        info = Bankinfo()
        acc = BankAccount()
        accType = input("enter your bank type:")
        if accType.lower() == "saving":
            sav = Saving()

        elif accType.lower() == "current":
            current = Current()


#




