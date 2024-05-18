
from __future__ import annotations
from typing import List, Generic, TypeVar



#created  an StackOverflowError exception class for data over flow using the BaseException class
class StackOverflowError(BaseException):
    pass
#created  an StackOverflowError exception class for data under flow using the BaseException class
class StackUnderflowError(BaseException):
    pass

T = TypeVar("T")
class Stack(Generic[T]):

    def __init__(self, limit: int = 10) -> None:
        self.Stack = []#internally using list for storing the elements
        self.limit = limit #setting the limit for the stack

    '''def __bool__(self) -> bool:
        return bool(self.Stack)
'''
    def __str__(self) -> str:
        return str(self.Stack) #this will return the stack in a string format
    
    #PUSH operation
    def push(self, element:T):
        if len(self.Stack) >= self.limit: # if size of stack crosses the limit the we cant add the elements into it
            raise StackOverflowError
        self.Stack.append(element)

    def pop(self):
        if (not self.Stack): #if empty we cant remove the element from the stack
            raise StackUnderflowError
        return self.Stack.pop() #this is the lists pop operation
    
    def peek(self):
        if (not self.Stack): #if empty we cant remove the element from the stack
            raise StackUnderflowError
        return self.Stack[-1] #returning the last element from list that means first element from the stack
    
    def is_empty(self):

        return not bool(self.Stack) # bool of empty stack is False and not of False is True
    
    def is_full(self): # checking if the stack is full or not

        return self.size() == self.limit
    
    def size(self): # size of the stack

        return len(self.Stack)
    
    def contains(self, item:T): # checking if the item is in the stack or not

        return item in self.Stack
   


stack: Stack[int] = Stack(10)
print(str(stack))

print(list(str(stack)))
print(bool(list(str(stack))))
print(bool(stack))
assert bool(stack) is True
assert stack.is_empty() is True
assert stack.is_full() is False
try:
    pop = stack.pop()
    raise AssertionError
except StackUnderflowError:
    assert True

try:
    peek = stack.peek()
    raise AssertionError
except StackUnderflowError:
    assert True
    

assert str(stack) == "[]"

for i in range(10):
        stack.push(i)

assert bool(stack)
assert not stack.is_empty()
assert stack.is_full()
assert str(stack) == str(list(range(10)))
assert stack.pop() == 9
assert stack.peek() == 8

stack.push(100)
assert str(stack) == str([0, 1, 2, 3, 4, 5, 6, 7, 8, 100])

try:
    stack.push(200)
    raise AssertionError  # This should not happen
except StackOverflowError:
    assert True  # This should happen

assert not stack.is_empty()
assert stack.size() == 10

assert stack.contains(5)
assert stack.contains(55) is False

