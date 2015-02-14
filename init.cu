#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "SpatialMaxPoolingPos.cu"
#include "SpatialMaxUnpoolingPos.cu"
#include "SpatialMlpUnPooling.cu"
#include "SpatialMlpPooling.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libjz(lua_State *L);

int luaopen_libjz(lua_State *L)
{
  lua_newtable(L);

  cunn_SpatialMaxPoolingPos_init(L);
  cunn_SpatialMaxUnpoolingPos_init(L);
  cunn_SpatialMlpUnPooling_init(L);
  cunn_SpatialMlpPooling_init(L);


  return 1;
}
