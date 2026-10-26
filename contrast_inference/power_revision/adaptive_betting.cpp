#include <algorithm>
#include <cmath>
#include <cstdint>
extern "C" int adaptive_betting(int N,int m,int K,const int8_t* H,
 const int32_t* a,const int32_t* b,double* result){
 if(N<1||m<1||K<1)return 1;
 for(int j=0;j<K;j++){
  double s=0,q=4,lp=0,lm=0;
  for(int k=0;k<N;k++){
   const double z=(H[(size_t)k*m+a[j]]-H[(size_t)k*m+b[j]])/2.0;
   // Stakes are functions only of pairs preceding the current pair.
   const double plus=std::clamp(s/q,0.0,0.5);
   const double minus=std::clamp(-s/q,0.0,0.5);
   lp+=std::log1p(plus*z);lm+=std::log1p(-minus*z);
   s+=z;q+=z*z;
  }
  result[j]=lp;result[K+j]=lm;
 }
 return 0;
}
