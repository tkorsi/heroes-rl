
#include <metal_stdlib>
using namespace metal;
kernel void pooled_forward(device const float* table,device const long* children,device const long* tags,device const long* offsets,device const long* counts,
 device const float* scale,device const float* w,device const float* bias,device float* out,device float* inp,
 constant uint& H,uint b [[threadgroup_position_in_grid]],uint j [[thread_index_in_threadgroup]]){
 threadgroup float x[64];float pool=0;uint K=uint(counts[b]);uint start=uint(offsets[b]);
 for(uint k=0;k<K;k++)pool+=table[children[start+k]*H+j];
 if(K)pool/=scale[b];
 x[j]=table[tags[b]*H+j];x[H+j]=pool;
 inp[b*2*H+j]=x[j];inp[b*2*H+H+j]=pool;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float v=bias[j];for(uint k=0;k<2*H;k++)v+=w[j*2*H+k]*x[k];out[b*H+j]=tanh(v);
}
kernel void pooled_backward(device atomic_float* arena,device const float* incoming,device const float* out,
 device const long* children,device const long* tags,device const long* offsets,device const long* counts,device const float* scale,device const float* w,device float* delta,
 constant uint& H,uint b [[threadgroup_position_in_grid]],uint j [[thread_index_in_threadgroup]]){
 uint K=uint(counts[b]);uint start=uint(offsets[b]);threadgroup float d[32];float y=out[b*H+j];d[j]=incoming[b*H+j]*(1-y*y);delta[b*H+j]=d[j];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float a=0,c=0;for(uint k=0;k<H;k++){a+=d[k]*w[k*2*H+j];c+=d[k]*w[k*2*H+H+j];}
 atomic_fetch_add_explicit(&arena[tags[b]*H+j],a,memory_order_relaxed);
 if(K){c/=scale[b];for(uint k=0;k<K;k++)atomic_fetch_add_explicit(&arena[children[start+k]*H+j],c,memory_order_relaxed);}
}

// Text padding row zero never receives a derivative.
kernel void embedding_scatter(device atomic_float* target,
 device const long* ids,device const float* values,
 constant uint& width,constant uint& rows,uint index [[thread_position_in_grid]]) {
 if(index>=rows*width)return;
 uint row=index/width,column=index%width;long id=ids[row];
 if(id!=0)atomic_fetch_add_explicit(&target[id*width+column],values[index],memory_order_relaxed);
}

// Each compiled DAG group contains unique destination rows.
kernel void write_rows(device float* table,device const long* rows,
 device const float* values,constant uint& width,constant uint& count,
 uint index [[thread_position_in_grid]]) {
 if(index<count*width)table[rows[index/width]*width+index%width]=values[index];
}
