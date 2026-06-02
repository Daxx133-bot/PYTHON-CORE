ch = input("enter a character: ")
if ch == 'a':
    print("you have entered a")
elif ch == 'b':
    print("you have entered b")
elif ch == 'c':
    print("you have entered c")
else:
    print("invalid character")

n = int(input("enter a number"))
if n > 0:
    if n%2 != 0:
        print(n+10)
    elif n%2 == 0:
        print(n*10.5)
elif n < 0:
    if n%2 != 0:
        print(n - 10)
    elif n%2 == 0:
        print(n/10.3)


marks = int(input("enter marks: "))
if marks >= 90 & marks <100:
    print("A grade")
elif marks >= 80 & marks < 90:
    print("B grade")
elif marks >= 60 & marks < 80:
    print("C grade")
elif  marks >= 40 & marks < 60:
    print("D grade")
elif marks < 40:
    print("fail")
elif marks >100:
    print("invalid marks")







