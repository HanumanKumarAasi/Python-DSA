def search(nums: [int], target: int):
    # write your code logic !!
    #Binary Search algorithm
    #iterative approach
    n = len(nums)

    low =  0
    high = n-1
    while(low<=high):
        
        mid = int(low+(high-low)/2)

        if nums[mid]==target : return mid

        if nums[mid]<target : low=mid+1

        if nums[mid]>target : high=mid-1

    return -1


#binary search recursive approach
def binaryReursive(nums: [int], low:int, high:int, target: int):
    if(low<=high):
        mid=int(low+(high-low)/2)

        if(nums[mid]==target): return mid

        if(nums[mid]<target): return binaryReursive(nums,mid+1,high,target)

        if(nums[mid]>target): return binaryReursive(nums,low,mid-1,target)
    
    return -1


#lower bound index concept

'''
Lower bound = arr[mid] >= x (smallest index such hat arr[index]>=target)
'''
def lowerBound(arr: [int], n: int, x: int) -> int:
    # initialise left and right end points
    l = 0
    r = n - 1

    # initialise lower_b with length of the array, because for an element which doesn't exist in the array,
    # it must have been existed after the last element of the array
    lower_b = n

    # loop until l <= r
    while l <= r:
        # get the mid index 
        mid = l + (r - l) // 2

        # if the mid element is greatest than or equal to the element x, 
        # update lower_b with mid index and search in the left half,
        # because we want an element just greater than or equal to not always greater than x
        if arr[mid] >= x: # search in the left half
            r = mid -1
            lower_b = min(mid, lower_b)

        else: # else search in the right half
            l = mid + 1

    return lower_b

'''
Upper bound = arr[mid] > target (smallest index such hat arr[index]>target)
'''
def upperBound(arr: [int], n: int, x: int) -> int:
    # Write your code here.

    ans = n
    low=0
    high=n-1

    while(low<=high):

        mid=low+(high-low)//2

        if(arr[mid]>x):
            ans=mid
            high=mid-1 #there might be getting small index element that is greater than x
        else:
            low=mid+1

    return ans

#search for correct index that target can be placed in(performed using lower bound)
def searchInsert(arr: [int], m: int) -> int:
    # Write your code here.

    n=len(arr)
    ans=n
    low=0
    high =n-1

    while(low<=high):
        mid=low+(high-low)//2
        if(arr[mid]>=m):
            ans=mid
            high=mid-1
        else:
            low=mid+1
    return ans

#floor and ceil of a target element
def getFloorAndCeil(a, n, x):
    # Write your code here.
    floor=-1
    ceil=-1
    ans=n
    low=0
    high=n-1
    while(low<=high): #3 4 7 8 8 10 x=5
        mid=low+(high-low)//2

        #finding the lower bound
        if(a[mid]==x):
            return(x,x)
        if(a[mid]>x):
            ceil=a[mid]
            high=mid-1
        else:
            floor=a[mid]
            low=mid+1
    
    return(floor,ceil)

#search first and last occurence
def firstAndLastPosition(arr, n, k):

    #this can be performed by using 2 approaches

    #first approach - lower bound and upper bound approach
    #second approach is written in notes i.e is normal BS operation for first occurence separate and last occurence separate.
    #first occurence = lower bound value and last occurence would be the upper bound value -1
    firstOccurence=lower_bound(arr,n,k)
    lastOccurence=upper_bound(arr,n,k)-1

    #if in case element doesnt exist in the search space so we will be returing -1 for first and last occurence
    if(firstOccurence==-1 or arr[firstOccurence] != k or firstOccurence==n):
        firstOccurence=-1
        lastOccurence=-1

    return(firstOccurence,lastOccurence)

def lower_bound(arr,n,k):
    first=-1
    l=0
    h=n-1
    while(l<=h):
        mid = l+(h-l)//2

        if(arr[mid]>=k):
            first=mid
            h=mid-1
        else:
            l=mid+1

    return first

def upper_bound(arr,n,k):
    last=n
    l=0
    h=n-1
    while(l<=h):
        mid = l+(h-l)//2

        if(arr[mid]>k):
            last=mid
            h=mid-1
        else:
            l=mid+1

    return last
#count no of occurence of target in the sorted array
def count(arr: [int], n: int, x: int) -> int:
    firstOccurence=lower_bound(arr,n,x)
    lastOccurence=upper_bound(arr,n,x)-1

    if(firstOccurence==-1 or arr[firstOccurence] != x or firstOccurence==n):
        firstOccurence=-1
        lastOccurence=-1

    if(firstOccurence==-1):
        return 0
    
    return lastOccurence-firstOccurence+1

#search the target in rotated sorted array which contains unique elements
def search(arr, n, k):

    l=0
    h=n-1

    while(l<=h):

        mid = l+(h-l)//2

        if(arr[mid]==k):
            return mid

        #need to check the sorted array either of left array or right array will be sorted for sure
        #checking for left array is sorted or not
        if(arr[l]<=arr[mid]):
            #if yes then we need to check whether the target exist in left sorted array or not
            if(arr[l]<=k and k<=arr[mid]):
                #if target exists in left sorted array then we will be ignoring right half array
                h=mid-1
            else:
                #else finding the target in the right half array
                l=mid+1
        else: #if left array is not sorted then right array may sorted and we need to find the array in right half array
            if(arr[mid]<=k and k<=arr[h]):
                #if target exists in right sorted array then we will be ignoring left half array
                l=mid+1
            else:
                #else finding the target in the right half array
                h=mid-1

    return -1

##search the target in rotated sorted array which contains duplicate elements
from typing import *

def searchInARotatedSortedArrayII(arr : List[int], key : int) -> bool:
    l=0
    h=len(arr)-1

    while(l<=h):

        mid = l+(h-l)//2

        if(arr[mid]==key):
            return True
        
        #we will shrinking the array until the sorted and unsorted part exists on either of the sides.
        if(arr[mid]==arr[l] and arr[mid]==arr[h]):
            l=l+1
            h=h-1
            continue

        #need to check the sorted array either of left array or right array will be sorted for sure
        #checking for left array is sorted or not
        if(arr[l]<=arr[mid]):
            #if yes then we need to check whether the target exist in left sorted array or not
            if(arr[l]<=key and key<=arr[mid]):
                #if target exists in left sorted array then we will be ignoring right half array
                h=mid-1
            else:
                #else finding the target in the right half array
                l=mid+1
        else: #if left array is not sorted then right array may sorted and we need to find the array in right half array
            if(arr[mid]<=key and key<=arr[h]):
                #if target exists in right sorted array then we will be ignoring left half array
                l=mid+1
            else:
                #else finding the target in the right half array
                h=mid-1

    return False


#checking for minimum element in roatated sorted array contains unique elements
#left sorted consider a[low] as min (or) right sorted consider a[mid] as min and check for ele in unsorted side may contains the min element
import sys

def findMin(arr: [int]):
    # Write your code here.
    mini = sys.maxsize

    l=0
    h=len(arr)-1

    while(l<=h):

        #we will be checking whether arr[l] is lesser than arr[h] in that case the array is sorted
        if(arr[l]<=arr[h]):
            mini = min(mini,arr[l])
            break

        mid=l+(h-l)//2
        #chceking if left array is sorted or not? if sorted assign consider arr[low] as minimum and try to check other element in right half array
        if(arr[l]<=arr[mid]):
            mini = min(mini,arr[l])
            l=mid+1 #need to check on right half array
        #if left array is not sorted then right array is sorted array so consider arr[mid] is minimum and try to check other element in left half array
        else:
            mini = min(mini,arr[mid])
            h=mid-1#need to check on left half array
            
    return mini


#finding the k rotations
#printing the index of minimum element is the no of rotations an array is rotated
def findKRotation(arr : [int]) -> int:
    # Write your code here.
    l =0
    h=len(arr)-1
    minIndex = 0

    while(l<=h):
        if(arr[l]<=arr[h]):
             minIndex = l if arr[minIndex]>arr[l] else minIndex
             break
        
        mid = l+(h-l)//2

        if(arr[l]<=arr[mid]):
            minIndex = l if arr[minIndex]>arr[l] else minIndex
            l=mid+1
        else:
            minIndex= mid if arr[minIndex]>arr[mid] else minIndex
            h=mid-1

    return minIndex

#priniting Non duplicate element in an sorted array
#even-odd/odd-even mechanism
def singleNonDuplicate(arr):
    # Write your code here
    l =0
    h=len(arr)-1

    #base conditions
    if(len(arr)==1):
        return arr[0]

    if(arr[l]!=arr[l+1]):
        return arr[0]
    if(arr[h]!=arr[h-1]):
        return arr[h]

    while(l<=h):

        mid = l+(h-l)//2
        if(arr[mid]!=arr[mid+1] and arr[mid]!=arr[mid-1]):
            return arr[mid]
        if((mid%2==1 and arr[mid]==arr[mid-1]) or (mid%2==0 and arr[mid]==arr[mid+1])):
            #this is for even odd even odd case so we need to check on right side
            l = mid+1
        else: # this is for odd even odd even case so we need to check on left side
            h = mid-1

    return -1

print(singleNonDuplicate([10]))

#searching the peak element in an array 
#-infi/arr[index-1]<arr[index]>arr[index+1]/-inif  mechanism
def findPeakElement(arr: [int]) -> int:
    # Write your code here

    n = len(arr)

    #some base executions if n==1 peak is 0 index
    if(n==1):
        return 0
    if(arr[0]>arr[1]): #if -infi<a[0]>a[1] so ) is peak
        return 0
    if(arr[n-1]>arr[n-2]): #a[n-2] < a[n-1] > -inif so n-1 is the peak
        return n-1

    l=1
    h=n-2

    while(l<=h):

        mid = l+(h-l)//2
        #if arr[mid-1]<arr[mid]>arr[mid+1] mid index ele is the peak ele
        if(arr[mid]>arr[mid-1] and arr[mid]>arr[mid+1]):
            return mid
        elif(arr[mid]>arr[mid-1]): #gradually increasining in left side so peak exists in right side
            l=mid+1 
        elif(arr[mid]>arr[mid+1]): #gradually decreasining in right side so peak exists in left side
            h=mid-1
        else:
            l=mid+1 #neither of the above cases executes that means we have multiple peaks in the array so either of the peak can be returned would be fine
                         #h=mid-1 would also be fine we can search peak in left side or l=mid+1 means searching for peak in right side

    return -1

#--------------------------------------BS on answers concepts---------------------------------------------------------------------------------------------------------------------------

#find the floor for the sqrt of the given number(the largest/maximum number i.e < n)
def floorSqrt(n):
   # write your code logic here .
   l=1
   h=n
   ans=-1 #considering ans as -1 initially
   while(l<=h):
      mid = l+(h-l)//2

      if(mid*mid <= n): #if mid*mid <= n then there may be value on right side to get the correct value or mid may be the correct value so assignthe mid to the ans and find out if any better ans will be find out on right side
         ans=mid
         l=mid+1
      else: #if mid*mid>n then we need find in left side
         h=mid-1
   return ans
'''n= int(input())
print(floorSqrt(n))'''
#finding the nth root for the given number m and returning the root value if exists else -1

def NthRoot(n: int, m: int) -> int:
    # Write Your Code Here

    l=1
    h=m

    while(l<=h):
        mid = l+(h-l)//2

        if(pow(mid,n)==m): return mid #checking if the pow(mid,n) == m then ans is mid

        if(pow(mid,n)>m): #checking if the pow(mid,n) > m then ans can be find out in left side
            h=mid-1
        else:
            l=mid+1 #checking if the pow(mid,n) < m then ans can find out in right side

    return -1# if no nth root exists...



import math
def findMax(v):
    maxi = float('-inf')
    n = len(v)
    # Find the maximum
    for i in range(n):
        maxi = max(maxi, v[i])
    return maxi

def findmin(v):
    mini = sys.maxsize
    n = len(v)
    # Find the maximum
    for i in range(n):
        mini = min(mini, v[i])
    return mini

def calculateTotalHours(v, hourly):
    #This function is used to calculate the no.of hours can be taken by eating hourly(no of bananas) with in an hour.
    totalH = 0
    n = len(v)
    # Find total hours
    for i in range(n):
        totalH += math.ceil(v[i] / hourly)
    return totalH

#koko banana problem concept - we need to find how may bananas that koko caan eat in an hour so that it can complete all the bananas from the given piles with in the given amount of time(here 8hrs is the amount of time they have given to us)
def minimumRateToEatBananas(v, h):
    low = 1
    high = findMax(v) #from 1 to max(v) the minimum value will exists
    ans=-1
    # Apply binary search
    while low <= high:
        mid = (low + high) // 2
        totalH = calculateTotalHours(v, mid) 
        if totalH <= h:
            ans=mid
            high = mid - 1
        else:
            low = mid + 1
    return ans

#bloomed adjacent flower bouquet concept
def roseGarden(arr: List[int], k: int, m: int):
    # write yur code here
    l=findmin(arr) #minimum num of days possible in bloomed array
    h=findMax(arr) #maximum num of days possible in bloomed array 
    ans=-1 #ans 

    if(m*k > len(arr)): return -1 #if number of flowers is lesser than no of flowers to make m bouquets then we cant make bouquest and will be returning -1 

    while(l<=h):
        mid=l+(h-l)//2

        if(bouqetPossibility(arr,mid,k)>=m): #if this condition is true consider mid as possible minimum no of days and check in left side for better possibility
            ans=mid
            h=mid-1
        else: #else check on right side for better possibility.
            l=mid+1
        
    return ans

def bouqetPossibility(arr:List[int],mid:int,k:int):
    count=0 #count for adjacent bloomed flowers for making bouqets
    noOfBouquet=0 #no of bouqets made using adjacent bloomed flowers
    for i in arr:
        if(i<=mid): #if i <= mid(which is actual bloomed number)  then count of bloomed flowers will be increased by 1
            count+=1
        else: #else we will be calculating noOfbouquest can be made with the existing(count) adjacent bloomed flower
            noOfBouquet+=count//k
            count=0
        
    noOfBouquet+=count//k #this can be calculated after the loop termination that count may have the value in it
    return noOfBouquet

#find the Smallest(minimum) divisor that provides the threshold <= given limit.
def smallestDivisor(arr: [int], limit: int) -> int:
    # Write your code here.
    l=1
    h=findMax(arr)
    divisor=-1
    while(l<=h):

        mid=l+(h-l)//2

        if(ceilDivisorFunction(arr,mid)<=limit): #checking the actualThreshold with the limit if this condition get statisfied we need check in the left side for minimum possible divisor that can match this condition
            divisor=mid
            h=mid-1
        else: #else we will be checking in right side for possible divisor
            l=mid+1

    return divisor




    pass

def ceilDivisorFunction(arr:[int],mid:int):
    actualThreshold = 0
    for i in arr:
        actualThreshold+=math.ceil(i/mid) #incrementing the ceil threshold for every element with respective divisor(mid)
    return actualThreshold

from os import *
from sys import *
from collections import *
from math import *
#finding the least weight capacity that we can ship in the given number of days
def leastWeightCapacity(weights, d):
    # Write your code here.

    low=findMinAndMax(weights)[0] # low is always should be max beacuse if we select low lesser than max we cant ship the max capacity so the least capacity will start from the max value in the weights
    high=findMinAndMax(weights)[1] # high capacity will be at most the sum of the weights that can be shifted in one day
    ans=-1#answer
    while(low<=high):
        mid=low+(high-low)//2
        noOfDay= findTheNoOfDaysForTheGivenCapacity(mid,weights) #calculating no of days that a capacity can take up to ship all the weights
        if(noOfDay<=d): # if noofdays is <= expected days then we can reduce the capacity because we need least capacity as anaswer
            ans=mid
            high=mid-1
        else: # else we need find the higher capacity on the right side
            low=mid+1

    return ans #returning the answer

def findMinAndMax(arr:[int]): #finding the max and min values using one single function
    low=float('-inf')
    high=0
    for i in arr:
        low = max(i,low)
        high = high+i
    return (low,high)


def findTheNoOfDaysForTheGivenCapacity(cap,weights):
    days=1
    load=0
    for i in weights:
        if(load+i > cap):
            days=days+1
            load=i
        else:
            load+=i
    return days


from typing import *
#finding the kth missing number in an sorted array
def missingK(vec: List[int], n: int, k: int) -> int:
    # Write your code here.
    low=0 #starting index
    high=n-1 #ending index

    while(low<=high):
        mid=low+(high-low)//2
        missingNoOfNumbers = vec[mid]-(mid+1) # finding the missing no of number wrt to that index using that index value and it wont be less than zero because it is an sorted array started from 0
        if(missingNoOfNumbers<k): low=mid+1 # if missingNumbers < k then we need find the K th missing number from the right side because we cant find kth missing number in the mssing number that are lesser than kth position
        else: high=mid-1 #else we can find in left side

    return high+1+k # this is the formula for returning the kth missing number

#aggressive Cows minimum of maximum distance that we can put them in the stalls
def aggressiveCows(stalls, k):
    n = len(stalls)  # size of array
    stalls.sort()  # sort the stalls

    low = 1
    high = stalls[n - 1] - stalls[0]
    # apply binary search
    while low <= high:
        mid = (low + high) // 2
        if canWePlace(stalls, mid, k):
            low = mid + 1
        else:
            high = mid - 1
    return high
def canWePlace(stalls, dist, cows):
    n = len(stalls)  # size of array
    cntCows = 1  # no. of cows placed
    last = stalls[0]  # position of last placed cow
    for i in range(1, n):
        if stalls[i] - last >= dist:
            cntCows += 1  # place next cow
            last = stalls[i]  # update the last location
        if cntCows >= cows:
            return True
    return False

import array
#Allocation of book, splitting of an array and finding minimum of largest sum of sub arrays and painters partition problems will be using the same concepts
def findPages(arr: [int], n: int, m: int) -> int:
    if m>n: #if no of students is greater than no of books present then return -1
        return -1
    low = findMax(arr) # this would be the minimum that we can assign to the student not less than this
    high = sum(arr) # this is the max that we can assign it to the student not more than this
    ans=-1
    while(low<=high):
        mid=low+(high-low)//2

        if(NoOfStudents(arr,mid)<=m): #if no of students is lesser then we can reduce the no of pages to students to alocate to all the student
            ans=mid
            high=mid-1
        else:
            low=mid+1
    return ans

def NoOfStudents(arr,maxPages): #functon for number students taken up those set of books
    student=1 #initially assigning the student 1
    studentPages=0 #num of pages assigned to the 1st student is 0 initially
    for i in range(len(arr)):
        if(studentPages+arr[i]<=maxPages): #if maxPages is lesser than the studentPages then he can take up somemore pages
            studentPages+=arr[i]
        else: #else we need assign next book pages to the next student
            student+=1 #next student
            studentPages=arr[i] #next book

    return student


def howMany(mid, arr):
    count = 0
    for i in range(len(arr) - 1):
        gap = arr[i + 1] - arr[i]
        count += (gap // mid)
    return count
#practice more examples
def minimiseMaxDistance(arr: [int], k: int) -> float:
    low = 0
    high = -1
    for i in range(len(arr) - 1):
        high = max(high, arr[i + 1] - arr[i]) #max consecutive distance between the two elemnent in the array
    
    while (high - low) >= 1e-7: #this is the decimal point that ans should not exceed beyond this
        mid = (low + high) / 2
        count = howMany(mid, arr)
        if count > k:
            low = mid
        else:
            high = mid
    return low

#better approach using count index tracker methodology for finding the median 
def median(a: int, b: int) -> float:
    # Write the function here.
    n=len(a)+len(b) #calculating the total length
    ind2 = n//2
    ind1 = ind2 -1 # this can be used for even indexes
    ind1ele = -1
    ind2ele = -2

    i=0 #start index for arr1
    j=0 #start index for arr2

    count=0 #for trackingup current index

    while(i<len(a) and j<len(b)):
        if(a[i]<b[j]):
            if(count==ind1): ind1ele=a[i]
            if(count==ind2): ind2ele=a[i]
            count+=1
            i+=1
        else:
            if(count==ind1): ind1ele=b[j]
            if(count==ind2): ind2ele=b[j]
            count+=1
            j+=1

    while(i<len(a)):
        if(count==ind1): ind1ele=a[i]
        if(count==ind2): ind2ele=a[i]
        i+=1
    while(j<len(b)):
        if(count==ind1): ind1ele=b[j]
        if(count==ind2): ind2ele=b[j]
        j+=1
    if(n%2==0):
        return (ind1ele+ind2ele)/2
    else:
        return ind2ele/1
    return -1
    
#BS approach for finding the median of the 2 sorted arrays using left part and right part methodology #always cconsider mimimum length array for BS operations
import sys
def median(a: int, b: int) -> float:
    # Write the function here.
    n1=len(a) #length of a
    n2=len(b) #length of b
    #always consider BS operation on min array length
    if(n1>n2): return median(b,a)

    l1= -sys. maxsize-1
    l2=  -sys. maxsize-1
    r1=sys.maxsize
    r2=sys.maxsize

    low=0
    high=n1 #length og minimum elements array
    leftpart = (n1+n2+1)//2
    while(low<=high):
        mid1=low+(high-low)//2
        mid2=leftpart-mid1

        if(mid1<n1): r1=a[mid1] #index needs to be checked else indexOutbound exceptions will betriggered
        if(mid2<n2): r2=b[mid2]
        if(mid1-1>=0): l1=a[mid1-1]
        if(mid2-1>=0): l2=b[mid2-1]

        if(l1<=r2 and l2<=r1):
            if((n1+n2)%2==1): return float(max(l1,l2))
            else: return float((max(l1,l2)+min(r1,r2))/2)

        if(l1>r2): high=mid1-1
        if(l2>r1): low=mid1+1

#find the kth element from the 2 sorted arrays
import sys
def kthElement(a: [int], n1: int, b: [int], n2: int, k: int) -> int:
    if(n1>n2): return kthElement(b,n2,a,n1,k)

    

    low=max(0,k-n2) #lets say if you have k>length of smaller array then you will lagged with the k-n1 elements then you will be lagged with the elements which shouldnt happen beacuse left part should contain k elements for sure then only you can abale to find out the kth index element else u wont
    high=min(k,n1) #min(length of minimum elements array, k) #atmost n1 if k<n1 then no need to use of other elements in the lower length array bcoz we need only the kth element
    leftpart =k
    while(low<=high):
        mid1=low+(high-low)//2
        mid2=leftpart-mid1
        l1= -sys. maxsize-1
        l2=  -sys. maxsize-1
        r1=sys.maxsize
        r2=sys.maxsize
        if(mid1<n1): r1=a[mid1] #index needs to be checked else indexOutbound exceptions will betriggered
        if(mid2<n2): r2=b[mid2]
        if(mid1-1>=0): l1=a[mid1-1]
        if(mid2-1>=0): l2=b[mid2-1]

        if(l1<=r2 and l2<=r1):
            return (max(l1,l2))

        if(l1>r2): high=mid1-1
        if(l2>r1): low=mid1+1
    return 0
#------------------------------------------------------------BS on 2D arrays-------------------------------------------------------------
#Row with maximum no of 1's in the 2D array
def rowWithMax1s(arr:[int], n:int, m:int, ):

    index = -1
    count_max=0

    for i in range(n):

        count_ones = m-lower_bound(arr[i],m,1)

        if(count_ones>count_max):
            count_max = count_ones
            index = i

    return index

def lower_bound(arr:[int],m,target):

    low=0
    high = m-1
    ans=m
    while(low<=high):
        mid=low+(high-low)//2
        if(arr[mid]>=target):
            ans=mid
            high=mid-1
        else:
            low=mid+1
    return ans


#search for the target in the 2D matrix
def searchMatrix(mat: [[int]], target: int) -> bool:
    # Write your code here.
    n=len(mat)
    for i in range(n):
        m=len(mat[i])
        targetIndex = lower_bound(mat[i],m,target)
        if(targetIndex<m and mat[i][targetIndex]==target):
            return True
    
    return False

#search for the target in the 2D matrix return1 if present else 0
def searchElement(matrix : List[List[int]], target : int) -> int:
    # Write your code here.
    n=len(matrix)
    for i in range(n):
        m=len(matrix[i])
        targetIndex = lower_bound(matrix[i],m,target)
        if(targetIndex<m and matrix[i][targetIndex]==target):
            return 1
    
    return 0


def findPeakGrid(g: [[int]]) -> [int]:
    # Write your code here.
    rows=len(g)
    cols=len(g[0])

    low=0
    high=cols-1 #becuase we are making column operation and finding the mid wrt the colums

    while(low<=high):
        mid=low+(high-low)//2
        maxEleRowIndex = findMaxEleRowIndex(g,rows,mid) #mid=column that we making operation on
        leftElementTotheMaxColEle = g[maxEleRowIndex][mid-1] if(mid-1>=0) else -1
        rightElementTotheMaxColEle = g[maxEleRowIndex][mid+1] if(mid+1<cols) else -1

        if(g[maxEleRowIndex][mid]>leftElementTotheMaxColEle and g[maxEleRowIndex][mid]>rightElementTotheMaxColEle):
            return (maxEleRowIndex,mid)
        
        if(g[maxEleRowIndex][mid]<leftElementTotheMaxColEle):
            high=mid-1
        else:
            low=mid+1
        
    
    return (-1,-1)

def findMaxEleRowIndex(arr,rows,selectedcol):
    index=-1
    maxele=-1
    for i in range(rows):
        if(arr[i][selectedcol]>maxele):
            maxele=arr[i][selectedcol]
            index=i
    return index

#Median in a row-wise sorted Matrix
def median(matrix: [[int]], m: int, n: int) -> int:
    # Write your code here.

    low = findmin(matrix,0) # this will be thefirst element
    high = findmax(matrix,n-1) #this will be the last element
    median = n*m//2 #median index
    while(low<=high):
        mid=low+(high-low)//2
        smallerEquals = elementsLesserThanorEqualsMid(matrix,m,n,mid) #finding the smaller elements that are <= mid

        if(smallerEquals<=median): #comparing the return count with median if it is <= median then it will be on right side
            low=mid+1
        else: #else it will be on left side
            high=mid-1

    return low #because we always want the median index value


def elementsLesserThanorEqualsMid(matrix, rows, cols,target):
    count=0
    for i in range(rows):
        count=count+upper_bound(matrix[i],cols,target)
    return count


arr = [10,20,30,40,50,60,70,80,90]
print('lower bound of 35 is')
print(lowerBound(arr,len(arr),30))
print(upperBound(arr,len(arr),30))
print(getFloorAndCeil(arr,len(arr),91))
arr1= [1,3,3,5]
print(firstAndLastPosition(arr1,len(arr1),3))
print(count(arr1,len(arr1),2))
rotatedArray = [7,8,9,1,2,3,4,5,6]
print(search(rotatedArray,len(rotatedArray),4))
rotatedArray1 = [27,31,43,45,46,5,11,13,18,19,20]
print(findKRotation(rotatedArray1))

print("cube root of 27 is : "+str(NthRoot(3,27)))


kokoArray=[7,15,6,3] #7bananas-2h,15bananas-3h,6bananas-2h,3bananas-1h
print("koko array")
print(minimumRateToEatBananas(kokoArray,8))

bloomedArray = [1,2,1,2,7,2,2,3,1]
print("")
print(roseGarden(bloomedArray,3,2))



smallestDivisorArray = [1,2,6,4,5]
print("")
print(smallestDivisor(smallestDivisorArray,8))

print(1//0.5)





print(findPeakGrid([[1,2,3],[4,5,6],[7,8,9]]))