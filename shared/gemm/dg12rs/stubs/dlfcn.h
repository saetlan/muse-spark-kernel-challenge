#pragma once
#define RTLD_LAZY 1
#define RTLD_LOCAL 0
extern "C" void* dlopen(const char*, int);
extern "C" void* dlsym(void*, const char*);
