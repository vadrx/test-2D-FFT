//
//  HighResTimer.hpp
//  test-2D-FFT
//
//  Created by Anton Volkov on 05/04/2018.
//  Copyright © 2018 Anton Volkov. All rights reserved.
//

#ifndef HighResTimer_hpp
#define HighResTimer_hpp

#include <string>
#include <chrono>
#include <iostream>
#include <sstream>

class HighResTimer
{
public:
    std::string description_;
    
    HighResTimer(const std::string& description = "HighResTimer");
    void start();
    void stop();
    void reset(const std::string& description);
    void reset();
    double getMicroseconds() const;
    double getMilliseconds() const;
    double getSeconds() const;
    double getMinutes() const;
    double getHours() const;
    
    std::string report() const;
    std::string reportMicroseconds() const;
    std::string reportMilliseconds() const;
    std::string reportSeconds() const;
    std::string reportMinutes() const;
    std::string reportHours() const;
    
    void print() const { std::string msString = report(); printf("[TIMER] %s\n", msString.c_str()); }
    void printSeconds() const { std::string msString = reportSeconds(); printf("[TIMER] %s\n", msString.c_str()); }
    void printMilliseconds() const { std::string msString = reportMilliseconds(); printf("[TIMER] %s\n", msString.c_str()); }
    void printMicroseconds() const { std::string msString = reportMicroseconds(); printf("[TIMER] %s\n", msString.c_str()); }
    
private:
    std::chrono::high_resolution_clock clock_;
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;
    std::chrono::duration<double> elapsed_time_;
};

class ScopedTimer
{
public:
    HighResTimer hrt_;
    ScopedTimer(const std::string& description = "ScopedTimer");
    ~ScopedTimer();
};

#endif /* HighResTimer_hpp */
