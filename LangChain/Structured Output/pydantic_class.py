from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str
    email: EmailStr
    address: str
    age: int
    native_place: Optional[str] = None

    # field function allows us to give description, add extra info
    cgpa: float = Field(description='marks in percentage', 
                        default=100, le=100, ge=0)




student = {
    'name': 'sanket',
    'email': 'sankettgorey@gmail.com',
    'address': 'Pune',
    'age': '32',
    'cgpa': 32
}


student1 = Student(**student)

print(student1)

print()

print(student1.model_dump_json())