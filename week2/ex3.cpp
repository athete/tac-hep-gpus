#include <iostream>

enum Throws {ROCK, PAPER, SCISSORS};

int determine_winner(Throws shape1, Throws shape2)
{
    if (shape1 == shape2)
        return 0;

    if (shape1 == ROCK && shape2 == PAPER)
        return 2;
    if (shape1 == ROCK && shape2 == SCISSORS)
        return 1;
    if (shape1 == PAPER && shape2 == ROCK)
        return 1;
    if (shape1 == PAPER && shape2 == SCISSORS)
        return 2;
    if (shape1 == SCISSORS && shape2 == ROCK)
        return 2;
    if (shape1 == SCISSORS && shape2 == PAPER)
        return 1;
    else
    {
        std::cout << "Invalid Option\n";
        exit(0);
    }
}

int main(int argc, char const *argv[])
{
    int input1, input2;

    std::cout << "Allowed tokens: ROCK (0), PAPER (1), SCISSORS (2)" << std::endl;
    std::cout << "Player 1: ";
    std::cin >> input1; 
    std::cout << "Player 2: ";
    std::cin >> input2; 

    Throws player1 = static_cast<Throws>(input1);
    Throws player2 = static_cast<Throws>(input2);

    int winner = determine_winner(player1, player2);
    switch(winner)
    {
        case 1: 
            std::cout << "\nPlayer 1 wins!" << std::endl;
            break;
        case 2: 
            std::cout << "\nPlayer 2 wins!" << std::endl;
            break;
        default: 
            std::cout << "\nDraw." << std::endl;
    }
    return 0;
}