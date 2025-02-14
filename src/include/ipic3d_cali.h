#ifndef __Ipic3d_Cali_H__
#define __Ipic3d_Cali_H__

#if HAVE_CALIPER
#include <caliper/cali.h>
#else
#define CALI_MARK_BEGIN(name)
#define CALI_MARK_END(name)
#define CALI_MARK_LOOP_BEGIN(id, name)
#define CALI_MARK_ITERATION_BEGIN(id, val)
#define CALI_MARK_ITERATION_END(id)
#define CALI_MARK_LOOP_END(id)
#define CALI_CXX_MARK_FUNCTION
#define CALI_CXX_MARK_SCOPE(name)
#endif

#endif /* __Ipic3d_Cali_H__ */