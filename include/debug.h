/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta. 
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite  
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of 
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*******************************************************************************************
  debug.h  -  definitions for debug and diagnostics (written by Alec Johnson)
 ********************************************************************************************/
#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <cstdarg>
#include <cstdio>

#include "errors.h"

void fprintf_fileLine(FILE * fptr, const char *type, const char *func,
  const char *file, int line_number, const char *format, ...);

#define dprintf(args...) fprintf_fileLine(stdout, "DEBUG", __func__, __FILE__, __LINE__,## args)
#define dprint(var) printvar_fileLine(__func__, __FILE__,__LINE__,#var,var);
#define declare_dprintvar_fileLine(type) \
void printvar_fileLine(const char*,const char*,int,const char*,type);

declare_dprintvar_fileLine(int);
declare_dprintvar_fileLine(long long);
declare_dprintvar_fileLine(double);
declare_dprintvar_fileLine(const char *);
declare_dprintvar_fileLine(const void *);

#endif
