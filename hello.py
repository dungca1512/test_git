#function to check a number is prime or not

def is_prime(n):
    for i in range(2,n):
        if n%i==0:
            return False
    return True

#function to print all prime numbers in a range
def print_primes(n):
    for i in range(2,n):
        if is_prime(i):
            print(i)

print(is_prime(5))
print(print_primes(10))

#function to check a number is even or odd

def is_even(n):
    if n%2==0:
        return True
    else:
        return False
    
print(is_even(5))
