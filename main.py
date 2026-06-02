print("hello world")
print("hello my name is: %s my roll no is %d how are you %s"%("a",10,"shyam"))

a ="hello world"
print(a.swapcase())

s = "-"
seq = ("a","b","c")
print(s.join(seq))
print(str.split("hello world"))
print(str.upper("hello world"))

#task 1

str = "hello world this is me daksh"
print(str[::-1])

words = str.split()
print(" ".join(words[::-1]))


print("*".join(words))
new_words = str.split()

result = ""
for i in range(0, len(str)-1, 2):

    result += str[i+1] + s[i]

if len(str) % 2 != 0:

    result += str[-1]

print(result)


print(str.replace(" is"," was"))

