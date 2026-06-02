def calculate(*lists):

    combined = [item for lst in lists for item in lst]
    print("Combined List:", combined)
    if len(lists) == 1:
        print(combined)
    elif len(lists) == 2:
        print("Maximum:", max(combined))
        print("Minimum:", min(combined))
    elif len(lists) == 3:
        print("Sum:", sum(combined))
    else:
        square = lambda x: x * x
        squared_list = list(map(square, combined))
        odd_list = list(filter(lambda x: x % 2 != 0, combined))
        print("Squared List:", squared_list)
        print("Odd List:", odd_list)
n = int(input("Enter number of lists: "))

all_list = []
for i in range(n):
    size = int(input("Enter number of elements in list %d: " % (i+1)))
    lst = []
    for j in range(size):
        element = int(input("Enter element: "))
        lst.append(element)
    all_list.append(lst)
calculate(*all_list)