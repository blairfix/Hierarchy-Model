//File: change_dir.h
#ifndef CHANGE_DIR_H
#define CHANGE_DIR_H


#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <iostream>

#include <algorithm>
#include <string>


void change_directory(std::string rmv, std::string add){

    char*  d = get_current_dir_name();

    //std::cout << d << std::endl;

    std::string dir(d);
    dir.erase(dir.find(rmv));
    dir.append(add);
    const char* dir_new = dir.c_str();

    chdir(dir_new);

    //std::cout << dir_new << std::endl;

}



#endif
