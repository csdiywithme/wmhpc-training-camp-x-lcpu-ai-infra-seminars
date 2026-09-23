#define main original_main
#include "01_fragment_map.cu"
#undef main
int main(){
int ar=0,ac=0,br=0,bc=0;
for(int l=0;l<32;l++)for(int i=0;i<16;i++){ar+=a_row_of(l,i)!=A_POS[l*16+i]/32;ac+=a_col_of(l,i)!=A_POS[l*16+i]%32;}
for(int l=0;l<32;l++)for(int i=0;i<8;i++){br+=b_row_of(l,i)!=B_POS[l*8+i]/8;bc+=b_col_of(l,i)!=B_POS[l*8+i]%8;}
printf("coordinate mismatches: A.row=%d/512 A.col=%d/512 B.row=%d/256 B.col=%d/256\n",ar,ac,br,bc);
}
