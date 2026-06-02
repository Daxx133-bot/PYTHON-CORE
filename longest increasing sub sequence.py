arr = [1,2,3,12,5,7,4,5,6,7,8]

n = len(arr)

for i in range(n):
    for j in range(i):
        if arr[i] > arr[j]:
            