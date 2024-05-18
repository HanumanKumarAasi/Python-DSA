# Python3 code
from typing import List
from collections import deque

#checking if the brackets expressing is balanced or not
def checkBalancedParanthesis(string:str):
    stack=[]
    for i in range(len(string)):
        if(string[i]=='{' or string[i]=='(' or string[i]=='['):
            stack.append(string[i])
        else:
            if stack:
                if stack[-1]=='(' and string[i]==')':
                    stack.pop()
                elif stack[-1]=='[' and string[i]==']':
                    stack.pop()
                elif stack[-1]=='{' and string[i]=='}':
                    stack.pop()
                else:
                    break
            else: return "not balanced"
    if (len(stack)==0):
        return "is balanced"
    else:
        return "not balanced"

#reversing the string TC:O(N) SC: O(N)
def reverseString(string:str):
    stack=[]
    for char in string:
        stack.append(char)

    string=""
    while stack:
        string+=stack.pop()
    
    return string


""" find the next greatest element for each element in the given array: https://www.naukri.com/code360/problems/next-greater-element_670312?utm_source=striver&utm_medium=website&utm_campaign=a_zcoursetuf&leftPanelTabValue=SUBMISSION"""
def nextGreaterElement(arr: List[int], n: int) -> List[int]:
    # Write your code here.
    temp=arr
    stack = [] #creating the empty stack

    for i in range(0,n): #iterating through each element

        while stack and stack[-1].get("value") < temp[i]: #checking if the stack is empty and checking stack top is < temp[i](next element of top)

            top = stack.pop()

            temp[top.get("ind")] = temp[i] #if the condition is true replacing the top element with its next greatest element



        stack.append({"value":temp[i],"ind":i}) # adding the next element to the stack

    while(stack): #if there are elements pended in the stack then there are no greatest element for them in the given tempay so we need to assign the -1 to them
        top = stack.pop()
        temp[top.get("ind")] = -1

    return temp


""" find the next smallest element for each element in the given array"""
def nextSmallerElement(arr: List[int], n: int) -> List[int]:
    # Write your code here.
    temp=arr
    stack = [] #creating the empty stack

    for i in range(0,n): #iterating through each element

        while stack and stack[-1].get("value") > temp[i]: #checking if the stack is empty and checking stack top is > temp[i](next element of top)

            top = stack.pop()

            temp[top.get("ind")] = temp[i] #if the condition is true replacing the top element with its next greatest element



        stack.append({"value":temp[i],"ind":i}) # adding the next element to the stack

    while(stack): #if there are elements pended in the stack then there are no greatest element for them in the given tempay so we need to assign the -1 to them
        top = stack.pop()
        temp[top.get("ind")] = -1

    return temp

""" find out the immediate smaller element for each element and replace with that element else replace with -1
https://www.naukri.com/code360/problems/immediate-smaller-element-_1062597?utm_source=striver&utm_medium=website&utm_campaign=a_zcoursetuf&leftPanelTabValue=PROBLEM """
def immediateSmaller(arr: List[int]) -> None:
    # Write your code here
    n = len(arr)
    for i in range(0,n-1): #iterating through n-1 element

        if(arr[i]>arr[i+1]): #comparing with its next element and replacing with the next element if it is lesser than current element else replacing it with the -1
            arr[i]=arr[i+1]
        else:
            arr[i] = -1
    
    arr[n-1]=-1# there wont be any next element to its right for the last element so it will be -1 always

    return arr

#Find next Smaller of next Greater in an array : https://www.geeksforgeeks.org/find-next-smaller-next-greater-array/
def nextSmallerOfNextGreater(arr: List[int], n: int):
    temp1=arr.copy()
    arr1= nextGreaterElement(temp1,7)
    temp2 =arr.copy()
    arr2= nextSmallerElement(temp2,7)
    ans = [None]*n
    for i in range(n):
        if(arr1[i] != -1):
            ind = arr.index(arr1[i])
            ans[i] = arr2[ind]
        else:
            ans[i]=-1
    return ans

"""Next Greater Element II : https://www.naukri.com/code360/problems/next-greater-element-ii_6212757?utm_source=striver&utm_medium=website&utm_campaign=a_zcoursetuf&leftPanelTabValue=PROBLEM"""
from typing import List
def nextGreaterElementII(arr: List[int]) -> List[int]:
    # Write your code here.
    n = len(arr)
    s = []
    arr2 = arr.copy() #making deep copy of the given array
    
    for i in range(0,n):

        while s and s[-1].get("value")<arr[i]:
            top = s.pop()
            arr2[top.get("ind")] = arr[i]

        s.append({"value":arr[i],"ind":i})
    #till here we have find outed the next greater element consider it as linear list

    #the below iteration is to find out the next greatest elements that are remaining in the stack, bcoz they have told that it is the circular list iterating one time can find the greatest element for the remaining stack elements
    for i in range(0,arr.index(s[0].get("value"))+1): #we can iterate through the index of the last element in the stack because the last element of the stack is the greatest element in the array and top of the stack is the least elemnt in the stack

        while s and s[-1].get("value")<arr[i]: #checking for the next greatest element for the top element from the stack
            top = s.pop()
            arr2[top.get("ind")] = arr[i]

    for i in range(0,len(s)): #assinging the -1 for the other elements in the stack because there wont be any greater elements found in the above iteration
        top = s.pop()
        arr2[top.get("ind")] = -1
    
    return arr2 #returning the answer
    
"""converting postfix expression to prefix expression using stack"""
def postfixToPrefix(s: str) -> str:
    # Write your code here.
    stack = []

    for i in range(len(s)):
        if(isOperator(s[i])):
            operand1=stack[-1]
            stack.pop()
            operand2=stack[-1]
            stack.pop()
            newString =  s[i]+operand2+operand1
            stack.append(newString)
        else:
            stack.append(s[i])
    return stack[-1]

    
def isOperator(ch:str ):
    if(ch in "+-*/"): return True
    return False


def prefixToInfixConversion(s: str) -> str:
    # Write your code here.
    stack = []
    i = len(s)-1
    while(i>=0):
        if(isOperator(s[i])):
            operand1=stack[-1]
            stack.pop()
            operand2=stack[-1]
            stack.pop()
            newString =  '('+operand1+s[i]+operand2+')'
            stack.append(newString)
            i-=1
        else:
            stack.append(s[i])
            i-=1
    return stack[-1]


def postToInfix(s: str) -> str:
    # Write your code here.
    stack = []
    i = 0
    while(i<len(s)):
        if(isOperator(s[i])):
            operand1=stack[-1]
            stack.pop()
            operand2=stack[-1]
            stack.pop()
            newString =  '('+operand2+s[i]+operand1+')'
            stack.append(newString)
            i+=1
        else:
            stack.append(s[i])
            i+=1
    return stack[-1]

"<--- right operation --->"
def preToPost(s: str) -> str:
    i=len(s)-1
    stack=[]
    while(i>=0):
        if(isOperator(s[i])):
            operand1=stack[-1]
            stack.pop()
            operand2=stack[-1]
            stack.pop()
            newString =  operand1+operand2+s[i]
            stack.append(newString)
            i-=1
        else:
            stack.append(s[i])
            i-=1
    return stack[-1]
# print(nextGreaterElement([4, 8, 5, 2, 25], 5))
# print(nextSmallerElement([4, 8, 5, 2, 25], 5))
# print(immediateSmaller([4, 8, 5, 2, 25]))
#print(nextSmallerOfNextGreater([5, 1, 9, 2, 5, 1, 7],7))
#print(nextGreaterElementII([5,6,7,4,3,2,1]))

#print(checkBalancedParanthesis("[()]{}{[()()]()}")) #balanced brackets
#print(checkBalancedParanthesis("[[(]){]}")) #unbalanced brackets

#print(reverseString("abcdefghijklmnopqrstuvwxyz"))

print(prefixToInfixConversion("*-a/bc-/dkl"))
