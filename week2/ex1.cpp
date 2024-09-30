#include <iostream>

void swap(int &a, int &b)
{
    a = a * b;
    b = a / b;
    a = a / b;
}

int main(int argc, char const *argv[])
{
    int A[10], B[10];
    
    // Initialize A with multiples of 2 and B with multiples of 3
    for (int idx = 0; idx < 10; idx++)
    {
        A[idx] = 2 * (idx + 1);
        B[idx] = 3 * (idx + 1);
    }
    std::cout << "Before swapping: " << std::endl;
    std::cout << "A: ";
    for (int x : A)
        std::cout << x << " ";
    std::cout << "\nB: ";
    for (int x : B)
        std::cout << x << " ";
    
    // Run swap
    for (int idx = 0; idx < 10; idx++)
        swap(A[idx], B[idx]);

    std::cout << "\nAfter swapping: " << std::endl;
    std::cout << "A: ";
    for (int x : A)
        std::cout << x << " ";
    std::cout << "\nB: ";
    for (int x : B)
        std::cout << x << " ";
    std::cout << std::endl;
    return 0;
}