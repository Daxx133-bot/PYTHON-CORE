def reverse_string(s):
    words = s.split()
    reversed = words[::-1]
    print(" ".join(reversed))

reverse_string("hello world this is daksh")

def interchange_elements(s,a,b):
    temp = list(s)
    temp[a],temp[b] = temp[b], temp[a]
    return "".join(temp)

string = input("enter a string")
a = int(input("enter first number"))
b = int(input("enter second number"))
print(interchange_elements(string,a,b))

