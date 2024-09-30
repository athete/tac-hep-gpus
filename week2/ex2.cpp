#include <iostream>
#include <string>

struct TACHEPStudent 
{
    std::string name, email, username, experiment;

};

void print_struct(const TACHEPStudent& student)
{
    std::cout << "Name: " << student.name << std::endl;
    std::cout << "Username: " << student.username << std::endl;
    std::cout << "Email: " << student.email << std::endl;
    std::cout << "Experiment: " << student.experiment << std::endl << std::endl;
}

int main(int argc, char const *argv[])
{
    TACHEPStudent s1, s2;

    s1.name = "Lael Verace";
    s1.email = "lverace@wisc.edu";
    s1.username = "lverace";
    s1.experiment = "CMS";

    s2.name = "Ameya Thete";
    s2.email = "thete@wisc.edu";
    s2.username = "thete";
    s2.experiment = "CMS";

    print_struct(s1);
    print_struct(s2);
    return 0;
}
