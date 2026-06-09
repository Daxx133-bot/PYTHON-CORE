print("enter the character string separated by comma: ")
a = input()
values = a.split(",")
int_list = []
str_list = []
for item in values:
    if item.isdigit():
        int_list.append(int(item))
    elif item.isalpha():
        str_list.append(item)
print(int_list)
print(max(int_list))
print(str_list)
for item in str_list:
    print(item[::-1])
