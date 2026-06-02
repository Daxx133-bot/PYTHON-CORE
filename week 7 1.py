class MaxLimitExceeded(Exception):
    def __init__(self, message):
        super().__init__(message)


class HDFCBank:

    def __init__(self):
        self.__transaction_limit = 3
        self.__amount_limit = 20000
    def withdraw(self, amount):

        if self.__transaction_limit <= 0:
            raise MaxLimitExceeded("Transaction limit exceeded")

        if amount > self.__amount_limit:
            raise MaxLimitExceeded("Amount limit exceeded")
        self.__amount_limit -= amount
        self.__transaction_limit -= 1

        print("Transaction Successful")
        print("Remaining Amount Limit:", self.__amount_limit)
        print("Remaining Transaction Limit:", self.__transaction_limit)


class AXISBank:

    def __init__(self):
        self.__transaction_limit = 5
        self.__amount_limit = 30000
    def withdraw(self, amount):
        if self.__transaction_limit <= 0:
            raise MaxLimitExceeded("Transaction limit exceeded")

        if amount > self.__amount_limit:
            raise MaxLimitExceeded("Amount limit exceeded")
        self.__amount_limit -= amount
        self.__transaction_limit -= 1
        print("Transaction Successful")
        print("Remaining Amount Limit:", self.__amount_limit)
        print("Remaining Transaction Limit:", self.__transaction_limit)


class ATM:

    def inputAmount(self):
        choice = input("Enter Bank Name(HDFC/AXIS): ")

        if choice.upper() == "HDFC":
            bank = HDFCBank()

        elif choice.upper() == "AXIS":
            bank = AXISBank()
        else:
            print("Invalid Bank")
            return
        while True:
            try:
                amount = int(input("Enter Amount to Withdraw: "))
                bank.withdraw(amount)
            except MaxLimitExceeded as e:
                print(e)
                break
            ch = input("Next Transaction(yes/no): ")
            if ch.lower() != "yes":
                print("Process Terminated")
                break
a = ATM()
a.inputAmount()