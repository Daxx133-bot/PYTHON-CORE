class Student_info:
    def __init__(self, student_rollno, student_name):
        self.student_rollno = student_rollno
        self.student_name = student_name

class StudentMarks:
    def __init__(self, rollno, marks1,marks2,marks3):
        self.rollno = rollno
        self.marks1 = marks1
        self.marks2 = marks2
        self.marks3 = marks3

student_list = []
class Main:
    def display(self):
        n = int(input("enter the numver of student:"))
        for i in range(n):
            roll_no = int(input("enter roll no:"))
            name = input("enter name:")
            info = Student_info(roll_no, name)

            print("enter student marks:")
            marks1 = int(input("enter marks 1:"))
            marks2 = int(input("enter marks 2:"))
            marks3 = int(input("enter marks 3:"))

            marks = StudentMarks(roll_no, marks1,marks2,marks3)

            avg = (marks1+marks2+marks3)/3

            def calculate(self, avg):
                if avg >= 90 and avg <= 100:
                    grade = "A"
                elif avg >= 80 and avg < 90:
                    grade = "B"
                elif avg >= 60 and avg < 80:
                    grade = "C"
                elif avg >= 40 and avg < 60:
                    grade = "D"
                elif avg < 40:
                    grade = "F"
                else:
                    print("invalid marks")
                return grade

            print("student details")
            print("student name:",info.student_name)
            print("studentt roll no:",info.student_rollno)
            print("avg marks:",avg)
            print("grade:",calculate(self,avg))

            student_list.append(info.Student_info)


obj = Main()
obj.display()


