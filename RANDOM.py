print("try with multiple except")
try:
    a = int(input("Enter number: "))
    print(10 / a)
except ValueError:
    print("Invalid Input")
except ZeroDivisionError:
    print("Cannot Divide By Zero")

print("multiple try and single except block ")
try:
    a = int(input("Enter number: "))
except:
    print("Invalid Input")
try:
    b = 10/a
    print(b)
except:
    print("error in division")

print("nested try with nested except")
try:
    a = int(input("Enter number: "))
    try:
        b = int(input("Enter number: "))
        c = a/b
        print(c)
    except:
        print("error in division")
except:
    print("invalid input")


print("nested try and nested except and finally")
try:
    print("this si the outer try")
    try:
        print("this is the innner try")
    except:
        print("inner except")
    finally:
        print("this si the inner finally block")
except:
    print("this si the outer except block")
finally:
    print("final outer finally block")

print("multiple try and single except block and finally block")
try:
    a = int(input("Enter number: "))
except:
    print("Invalid Input")
try:
    a = int(input("Enter number: "))
except:
    print("Invalid Input")
finally:
    print("end of the input section ")

try:
    b = 10/a
except:
    print("error in division")
finally:
    print("end of division secion")




