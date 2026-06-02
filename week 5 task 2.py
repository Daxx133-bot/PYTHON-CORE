n = int(input("enter the number of entries you want to make:"))
student = {}
for i in range(n):
    key = "s" + str(i)
    rollno = int(input("enter the rollno:"))
    name = input("enter the name:")
    marks = int(input("enter the marks:"))
    grade = None

    student[key] = {
        "rollno": rollno,
        "name": name,
        "marks": marks,
        "grade": grade
    }
#item function

for key in student:
    marks = student[key]["marks"]

    if marks >= 90:
        student[key]["grade"] = "A"
    elif marks >= 80 & marks <= 90:
        student[key]["grade"] = "B"
    elif marks >= 60 & marks <= 80:
        student[key]["grade"] = "C"
    elif marks >= 40 & marks <= 60:
        student[key]["grade"] = "D"
    else:
        student[key]["grade"] = "fail"

for key in student:
    print(key , ":", student[key])

