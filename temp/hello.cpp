#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    json j;
    j["project"] = "FridgeAI";
    j["status"] = "Learning Docker";
    
    std::cout << j.dump(4) << std::endl;
    return 0;
}