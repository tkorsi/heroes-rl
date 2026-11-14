#include <metal_stdlib>
using namespace metal;
#pragma STDC FP_CONTRACT OFF
kernel void backward_gru(device const float* incoming,device const float* states,device const float* gates,
 device const float* wh,device float* dgi,device float* dgh,device float* dh0,
 constant uint& T,constant uint& H,uint b [[threadgroup_position_in_grid]],uint j [[thread_index_in_threadgroup]]) {
 threadgroup float dg[192];threadgroup float carry[64];carry[j]=0;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(int t=int(T)-1;t>=0;t--) {
  uint g=(b*T+uint(t))*4*H;float r=gates[g+j],z=gates[g+H+j],n=gates[g+2*H+j],an=gates[g+3*H+j];
  float dh=carry[j]+incoming[(b*T+uint(t))*H+j];
  float dn=dh*(1.0f-z)*(1.0f-n*n);
  float dz=dh*(states[(b*(T+1)+uint(t))*H+j]-n)*z*(1.0f-z);
  float dr=dn*an*r*(1.0f-r);
  uint o=(b*T+uint(t))*3*H;
  dgi[o+j]=dr;dgi[o+H+j]=dz;dgi[o+2*H+j]=dn;
  dg[j]=dr;dg[H+j]=dz;dg[2*H+j]=dn*r;
  dgh[o+j]=dr;dgh[o+H+j]=dz;dgh[o+2*H+j]=dn*r;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float prev=dh*z;
  for(uint k=0;k<3*H;k++)prev+=dg[k]*wh[k*H+j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  carry[j]=prev;
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 dh0[b*H+j]=carry[j];
}
kernel void forward_gru_simd(device const float* gi, device const float* wh, device const float* bh,
 device const float* initial, device float* states, device float* gates,
 constant uint& T, constant uint& H, uint b [[threadgroup_position_in_grid]],
 uint s [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
 threadgroup float hidden[64];uint j=2*s;
 if(lane==0){hidden[j]=initial[b*H+j];hidden[j+1]=initial[b*H+j+1];states[b*(T+1)*H+j]=hidden[j];states[b*(T+1)*H+j+1]=hidden[j+1];}
 float wr0[2],wz0[2],wn0[2],wr1[2],wz1[2],wn1[2];
 for(uint q=0;q<H/32;q++){uint k=q*32+lane;wr0[q]=wh[j*H+k];wz0[q]=wh[(H+j)*H+k];wn0[q]=wh[(2*H+j)*H+k];wr1[q]=wh[(j+1)*H+k];wz1[q]=wh[(H+j+1)*H+k];wn1[q]=wh[(2*H+j+1)*H+k];}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint t=0;t<T;t++){
  float ar0=0,az0=0,an0=0,ar1=0,az1=0,an1=0;
  for(uint q=0;q<H/32;q++){float h=hidden[q*32+lane];ar0+=wr0[q]*h;az0+=wz0[q]*h;an0+=wn0[q]*h;ar1+=wr1[q]*h;az1+=wz1[q]*h;an1+=wn1[q]*h;}
  ar0=simd_sum(ar0);az0=simd_sum(az0);an0=simd_sum(an0);ar1=simd_sum(ar1);az1=simd_sum(az1);an1=simd_sum(an1);
  float h0=0,h1=0;
  if(lane==0){
   uint base=(b*T+t)*3*H;uint g=(b*T+t)*4*H;
   an0+=bh[2*H+j];an1+=bh[2*H+j+1];
   float r0=1.0f/(1.0f+exp(-((ar0+bh[j])+gi[base+j]))),z0=1.0f/(1.0f+exp(-((az0+bh[H+j])+gi[base+H+j]))),n0=tanh(gi[base+2*H+j]+r0*an0);
   float r1=1.0f/(1.0f+exp(-((ar1+bh[j+1])+gi[base+j+1]))),z1=1.0f/(1.0f+exp(-((az1+bh[H+j+1])+gi[base+H+j+1]))),n1=tanh(gi[base+2*H+j+1]+r1*an1);
   h0=(hidden[j]-n0)*z0+n0;h1=(hidden[j+1]-n1)*z1+n1;
   gates[g+j]=r0;gates[g+H+j]=z0;gates[g+2*H+j]=n0;gates[g+3*H+j]=an0;
   gates[g+j+1]=r1;gates[g+H+j+1]=z1;gates[g+2*H+j+1]=n1;gates[g+3*H+j+1]=an1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0){hidden[j]=h0;hidden[j+1]=h1;states[(b*(T+1)+t+1)*H+j]=h0;states[(b*(T+1)+t+1)*H+j+1]=h1;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
}
kernel void forward_gru_r16(device const float* gi,device const float* wh,device const float* bh,device const float* initial,
 device float* states,device float* gates,constant uint& T,constant uint& H,uint b [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
 constexpr uint R=16;uint j=tid/R,lane=tid%R;threadgroup float hidden[64];
 if(lane==0){hidden[j]=initial[b*H+j];states[b*(T+1)*H+j]=hidden[j];}
 float wr[64/R],wz[64/R],wn[64/R];for(uint q=0;q<H/R;q++){uint k=q*R+lane;wr[q]=wh[j*H+k];wz[q]=wh[(H+j)*H+k];wn[q]=wh[(2*H+j)*H+k];}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint t=0;t<T;t++){
  float ar=0,az=0,an=0;for(uint q=0;q<H/R;q++){float h=hidden[q*R+lane];ar+=wr[q]*h;az+=wz[q]*h;an+=wn[q]*h;}
  for(uint offset=R/2;offset;offset/=2){ar+=simd_shuffle_xor(ar,offset);az+=simd_shuffle_xor(az,offset);an+=simd_shuffle_xor(an,offset);}
  float h=0;if(lane==0){uint i=(b*T+t)*3*H,g=(b*T+t)*4*H;an+=bh[2*H+j];float r=1/(1+exp(-((ar+bh[j])+gi[i+j]))),z=1/(1+exp(-((az+bh[H+j])+gi[i+H+j]))),n=tanh(gi[i+2*H+j]+r*an);h=(hidden[j]-n)*z+n;gates[g+j]=r;gates[g+H+j]=z;gates[g+2*H+j]=n;gates[g+3*H+j]=an;}
  threadgroup_barrier(mem_flags::mem_threadgroup);if(lane==0){hidden[j]=h;states[(b*(T+1)+t+1)*H+j]=h;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
}

// Length-aware variants are used only for padded text/ordered tree batches.
kernel void backward_gru_lengths(device const float* incoming,device const float* states,device const float* gates,
 device const float* wh,device float* dgi,device float* dgh,device float* dh0,
 device const long* lengths,constant uint& T,constant uint& H,uint b [[threadgroup_position_in_grid]],uint j [[thread_index_in_threadgroup]]) {
 threadgroup float dg[192];threadgroup float carry[64];carry[j]=0;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(int t=int(min(T,uint(lengths[b])))-1;t>=0;t--) {
  uint g=(b*T+uint(t))*4*H;float r=gates[g+j],z=gates[g+H+j],n=gates[g+2*H+j],an=gates[g+3*H+j];
  float dh=carry[j]+incoming[(b*T+uint(t))*H+j];
  float dn=dh*(1.0f-z)*(1.0f-n*n);
  float dz=dh*(states[(b*(T+1)+uint(t))*H+j]-n)*z*(1.0f-z);
  float dr=dn*an*r*(1.0f-r);
  uint o=(b*T+uint(t))*3*H;
  dgi[o+j]=dr;dgi[o+H+j]=dz;dgi[o+2*H+j]=dn;
  dg[j]=dr;dg[H+j]=dz;dg[2*H+j]=dn*r;
  dgh[o+j]=dr;dgh[o+H+j]=dz;dgh[o+2*H+j]=dn*r;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float prev=dh*z;
  for(uint k=0;k<3*H;k++)prev+=dg[k]*wh[k*H+j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  carry[j]=prev;
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 dh0[b*H+j]=carry[j];
}
kernel void forward_gru_simd_lengths(device const float* gi, device const float* wh, device const float* bh,
 device const float* initial, device float* states, device float* gates,
 device const long* lengths,constant uint& T, constant uint& H, uint b [[threadgroup_position_in_grid]],
 uint s [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
 threadgroup float hidden[64];uint j=2*s;
 if(lane==0){hidden[j]=initial[b*H+j];hidden[j+1]=initial[b*H+j+1];states[b*(T+1)*H+j]=hidden[j];states[b*(T+1)*H+j+1]=hidden[j+1];}
 float wr0[2],wz0[2],wn0[2],wr1[2],wz1[2],wn1[2];
 for(uint q=0;q<H/32;q++){uint k=q*32+lane;wr0[q]=wh[j*H+k];wz0[q]=wh[(H+j)*H+k];wn0[q]=wh[(2*H+j)*H+k];wr1[q]=wh[(j+1)*H+k];wz1[q]=wh[(H+j+1)*H+k];wn1[q]=wh[(2*H+j+1)*H+k];}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint t=0;t<min(T,uint(lengths[b]));t++){
  float ar0=0,az0=0,an0=0,ar1=0,az1=0,an1=0;
  for(uint q=0;q<H/32;q++){float h=hidden[q*32+lane];ar0+=wr0[q]*h;az0+=wz0[q]*h;an0+=wn0[q]*h;ar1+=wr1[q]*h;az1+=wz1[q]*h;an1+=wn1[q]*h;}
  ar0=simd_sum(ar0);az0=simd_sum(az0);an0=simd_sum(an0);ar1=simd_sum(ar1);az1=simd_sum(az1);an1=simd_sum(an1);
  float h0=0,h1=0;
  if(lane==0){
   uint base=(b*T+t)*3*H;uint g=(b*T+t)*4*H;
   an0+=bh[2*H+j];an1+=bh[2*H+j+1];
   float r0=1.0f/(1.0f+exp(-((ar0+bh[j])+gi[base+j]))),z0=1.0f/(1.0f+exp(-((az0+bh[H+j])+gi[base+H+j]))),n0=tanh(gi[base+2*H+j]+r0*an0);
   float r1=1.0f/(1.0f+exp(-((ar1+bh[j+1])+gi[base+j+1]))),z1=1.0f/(1.0f+exp(-((az1+bh[H+j+1])+gi[base+H+j+1]))),n1=tanh(gi[base+2*H+j+1]+r1*an1);
   h0=(hidden[j]-n0)*z0+n0;h1=(hidden[j+1]-n1)*z1+n1;
   gates[g+j]=r0;gates[g+H+j]=z0;gates[g+2*H+j]=n0;gates[g+3*H+j]=an0;
   gates[g+j+1]=r1;gates[g+H+j+1]=z1;gates[g+2*H+j+1]=n1;gates[g+3*H+j+1]=an1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0){hidden[j]=h0;hidden[j+1]=h1;states[(b*(T+1)+t+1)*H+j]=h0;states[(b*(T+1)+t+1)*H+j+1]=h1;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
}
kernel void forward_gru_r16_lengths(device const float* gi,device const float* wh,device const float* bh,device const float* initial,
 device float* states,device float* gates,device const long* lengths,constant uint& T,constant uint& H,uint b [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
 constexpr uint R=16;uint j=tid/R,lane=tid%R;threadgroup float hidden[64];
 if(lane==0){hidden[j]=initial[b*H+j];states[b*(T+1)*H+j]=hidden[j];}
 float wr[64/R],wz[64/R],wn[64/R];for(uint q=0;q<H/R;q++){uint k=q*R+lane;wr[q]=wh[j*H+k];wz[q]=wh[(H+j)*H+k];wn[q]=wh[(2*H+j)*H+k];}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint t=0;t<min(T,uint(lengths[b]));t++){
  float ar=0,az=0,an=0;for(uint q=0;q<H/R;q++){float h=hidden[q*R+lane];ar+=wr[q]*h;az+=wz[q]*h;an+=wn[q]*h;}
  for(uint offset=R/2;offset;offset/=2){ar+=simd_shuffle_xor(ar,offset);az+=simd_shuffle_xor(az,offset);an+=simd_shuffle_xor(an,offset);}
  float h=0;if(lane==0){uint i=(b*T+t)*3*H,g=(b*T+t)*4*H;an+=bh[2*H+j];float r=1/(1+exp(-((ar+bh[j])+gi[i+j]))),z=1/(1+exp(-((az+bh[H+j])+gi[i+H+j]))),n=tanh(gi[i+2*H+j]+r*an);h=(hidden[j]-n)*z+n;gates[g+j]=r;gates[g+H+j]=z;gates[g+2*H+j]=n;gates[g+3*H+j]=an;}
  threadgroup_barrier(mem_flags::mem_threadgroup);if(lane==0){hidden[j]=h;states[(b*(T+1)+t+1)*H+j]=h;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
}
