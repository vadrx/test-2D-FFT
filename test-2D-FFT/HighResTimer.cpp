//
//  HighResTimer.cpp
//  test-2D-FFT
//
//  Created by Anton Volkov on 05/04/2018.
//  Copyright © 2018 Anton Volkov. All rights reserved.
//

#include "HighResTimer.hpp"

HighResTimer::HighResTimer(const std::string& description): description_(description)
{
    
}

void HighResTimer::start()
{
    start_ = clock_.now();
}

void HighResTimer::stop()
{
    end_ = clock_.now();
    elapsed_time_ = end_ - start_;
}

void HighResTimer::reset(const std::string& description)
{
    description_ = description;
    elapsed_time_.zero();
}

void HighResTimer::reset()
{
    elapsed_time_.zero();
}

double HighResTimer::getMicroseconds() const
{
    return elapsed_time_.count() * 1000000.0;
}

double HighResTimer::getMilliseconds() const
{
    return elapsed_time_.count() * 1000.0;
}

double HighResTimer::getSeconds() const
{
    return elapsed_time_.count();
}

double HighResTimer::getMinutes() const
{
    return getSeconds() / 60.;
}

double HighResTimer::getHours() const
{
    return getMinutes() / 60.;
}

std::string HighResTimer::reportMicroseconds() const
{
    std::ostringstream oss; oss << description_ << ": " << getMicroseconds() << " microseconds.";
    return oss.str();
}

std::string HighResTimer::reportMilliseconds() const
{
    std::ostringstream oss; oss << description_ << ": " << getMilliseconds() << " milliseconds.";
    return oss.str();
}

std::string HighResTimer::reportSeconds() const
{
    std::ostringstream oss; oss << description_ << ": " << getSeconds() << " seconds.";
    return oss.str();
}

std::string HighResTimer::reportMinutes() const
{
    std::ostringstream oss; oss << description_ << ": " << getMinutes() << " minutes.";
    return oss.str();
}

std::string HighResTimer::reportHours() const
{
    std::ostringstream oss; oss << description_ << ": " << getHours() << " hours.";
    return oss.str();
}

std::string HighResTimer::report() const
{
    double val = getMicroseconds();
    if (val <= 1000.0)
        return reportMicroseconds();
    
    val /= 1000.0;
    if (val <= 1000.0 && val >= 1.0)
        return reportMilliseconds();
    
    val /= 1000.0;
    if (val <= 60.0 && val >= 1.0)
        return reportSeconds();
    
    val /= 60.0;
    if (val <= 60.0 && val >= 1.0)
        return reportMinutes();
    
    val /= 60.0;
    return reportHours();
}

ScopedTimer::ScopedTimer(const std::string& description) :
hrt_(description)
{
    hrt_.start();
}

ScopedTimer::~ScopedTimer()
{
    hrt_.stop();
    std::cout << hrt_.report() << std::endl;
}
