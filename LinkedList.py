class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Do not change code above.


def constructLL(arr: [int]) -> Node:
    # Write your code here
    head = Node(arr[0])
    mover = head

    for i in range(1,len(arr)):
        temp = Node(arr[i])
        mover.next = temp
        mover=mover.next
    return head

def length(head) :
    #Your code goes here
    temp=head
    len=0
    while(temp is not None):
        len+=1
        temp=temp.next
    return len

def searchInLinkedList(head, k):
    # Your code goes here.
    temp = head
    while(temp is not None):
        if(temp.data == k):
            return 1
        temp = temp.next
    return 0
