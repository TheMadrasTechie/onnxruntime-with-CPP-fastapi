#include <iostream>

extern "C" {
    void print_numbers() {
        for (int i = 1; i <= 1000000; ++i) {
            std::cout << i << std::endl;
        }
    }
}
