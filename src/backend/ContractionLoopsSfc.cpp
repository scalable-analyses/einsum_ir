#include "ContractionLoopsSfc.h"
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

void einsum_ir::backend::ContractionLoopsSFC::init( std::vector< dim_t >   const & i_loop_dim_type,
                                                    std::vector< exec_t >  const & i_loop_exec_type,
                                                    std::vector< int64_t > const & i_loop_sizes,
                                                    std::vector< int64_t > const & i_loop_strides_left,
                                                    std::vector< int64_t > const & i_loop_strides_right,
                                                    std::vector< int64_t > const & i_loop_strides_out_aux,
                                                    std::vector< int64_t > const & i_loop_strides_out){

  
}

einsum_ir::err_t einsum_ir::backend::ContractionLoopsSFC::compile(){
  
}





/*
OLD
*/
#include "ContractionLoops.h"
#include <algorithm>
#include <stdlib.h>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif


int gilbert_d2xy_r(int dst_idx, int cur_idx,
                       int *xres, int *yres,
                       int ax,int ay,
                       int bx,int by );

int gilbert_d2xy(int *x, int *y, int idx, int w,int h);

void convertStridesToOffsets( std::map< int64_t, int64_t > const * i_sizes,
                              std::map< int64_t, int64_t > const * i_strides,
                              std::vector< int64_t>        const * i_dim_ids,
                              std::vector< int64_t >             & io_offsets ) {
                                
  io_offsets.resize( 2 * i_dim_ids->size() + 1);
  io_offsets[0] = 0;

  int64_t l_size_all = 0;
  for( size_t l_di = 0;l_di < i_dim_ids->size(); l_di++){
    int64_t l_dim_id = i_dim_ids->at(i_dim_ids->size() - l_di - 1);
    int64_t l_size   = i_sizes->at(  l_dim_id);
    int64_t l_stride = i_strides->at(l_dim_id);

    io_offsets[ 2 * l_di + 1] = l_stride   - l_size_all;
    io_offsets[ 2 * l_di + 2] = l_size_all - l_stride;
    std::cout << l_dim_id << " " << io_offsets[ 2 * l_di + 1] <<  " " << l_stride << " " << l_size_all  <<  std::endl;
    std::cout << l_size << std::endl;
    l_size_all += (l_size - 1) * l_stride;
  }
}

int64_t getMaxDimJump( std::map< int64_t, int64_t > const * i_sizes,
                       std::vector< int64_t>        const * i_dim_ids,
                       int64_t                              i_id_new,
                       int64_t                              i_id_old ){

  int64_t l_dif = i_id_new - i_id_old;
  int64_t l_max_id = i_id_new > i_id_old ? i_id_new : i_id_old;
  for( size_t l_di = 0; l_di < i_dim_ids->size(); l_di++ ){
    int64_t l_dim_id = i_dim_ids->at(i_dim_ids->size() - l_di - 1);
    int64_t l_size = i_sizes->at(l_dim_id);
    if(l_max_id % l_size != 0){
      return (l_di + 1) * l_dif;
    }
    else{
      l_max_id /= l_size;
    }
  }

  return 0;
}

void einsum_ir::backend::ContractionLoops::init( std::map< int64_t, int64_t > const * i_sizes,
                                                 std::map< int64_t, int64_t > const * i_strides_left,
                                                 std::map< int64_t, int64_t > const * i_strides_right,
                                                 std::map< int64_t, int64_t > const * i_strides_out_aux,
                                                 std::map< int64_t, int64_t > const * i_strides_out,
                                                 std::map< int64_t, dim_t >   const * i_dim_type,
                                                 std::vector<int64_t>               * i_loop_ids,
                                                 int64_t                              i_num_bytes_scalar_left,
                                                 int64_t                              i_num_bytes_scalar_right,
                                                 int64_t                              i_num_bytes_scalar_out,
                                                 kernel_t                             i_ktype_first_touch,
                                                 kernel_t                             i_ktype_main,
                                                 kernel_t                             i_ktype_last_touch,
                                                 ContractionPackingTpp              * i_packing ) {
  m_sizes = i_sizes;

  m_strides_left = i_strides_left;
  m_strides_right = i_strides_right;
  m_strides_out_aux = i_strides_out_aux;
  m_strides_out = i_strides_out;

  m_loop_ids = i_loop_ids;

  m_num_bytes_scalar_left  = i_num_bytes_scalar_left;
  m_num_bytes_scalar_right = i_num_bytes_scalar_right;
  m_num_bytes_scalar_out   = i_num_bytes_scalar_out;

  m_ktype_first_touch = i_ktype_first_touch;
  m_ktype_main        = i_ktype_main;
  m_ktype_last_touch  = i_ktype_last_touch;

  m_num_tasks_targeted = 1;

  m_packing = i_packing;

  m_dim_type = i_dim_type;

  m_threading_first_last_touch = false;

  m_compiled = false;
}

einsum_ir::err_t einsum_ir::backend::ContractionLoops::compile() {
  // determine if the outermost C dimension is complex
  m_cpx_outer_c = false;
  m_cpx_outer_c |= ce_cpx_op( m_ktype_first_touch );
  m_cpx_outer_c |= ce_cpx_op( m_ktype_main );
  m_cpx_outer_c |= ce_cpx_op( m_ktype_last_touch );

  // check if a complex C dimension is possible
  if( m_cpx_outer_c && m_dim_type->at( m_loop_ids->at(0) ) != einsum_ir::C ) {
    return err_t::INVALID_CPX_DIM;
  }
  if( m_cpx_outer_c && m_sizes->at( m_loop_ids->at(0) ) != 2 ) {
    return err_t::INVALID_CPX_DIM;
  }

  // derive complex strides
  if( m_cpx_outer_c ) {
    m_cpx_stride_in_left_bytes  = m_strides_left->at( m_loop_ids->at(0) )    * m_num_bytes_scalar_left;
    m_cpx_stride_in_right_bytes = m_strides_right->at( m_loop_ids->at(0) )   * m_num_bytes_scalar_right;
    m_cpx_stride_out_aux_bytes  = m_strides_out_aux->at( m_loop_ids->at(0) ) * m_num_bytes_scalar_out;
    m_cpx_stride_out_bytes      = m_strides_out->at( m_loop_ids->at(0) )     * m_num_bytes_scalar_out;
    m_loop_ids->erase(m_loop_ids->begin());
  }
  else {
    m_cpx_stride_in_left_bytes  = 0;
    m_cpx_stride_in_right_bytes = 0;
    m_cpx_stride_out_aux_bytes  = 0;
    m_cpx_stride_out_bytes      = 0;
  }

  std::cout << "test" << m_loop_ids->size() << std::endl;
  //get m and n dimensions of sfc
  std::vector< int64_t > l_sfc_dims_m;
  std::vector< int64_t > l_sfc_dims_n;
  int64_t l_size_sfc_m = 1;
  int64_t l_size_sfc_n = 1;
  int64_t l_size_sfc   = 1;
  int64_t l_id_sfc = -1;
  for( int64_t l_di = 0; l_di < m_loop_ids->size(); l_di++ ){
    int64_t l_dim_id   = m_loop_ids->at(l_di);
    std::cout << l_dim_id << std::endl;
    int64_t l_dim_size = map_find_default<int64_t>( m_sizes,           l_dim_id, 1                    );
    dim_t     dim_type = map_find_default<dim_t  >( m_dim_type,        l_dim_id, dim_t::UNDEFINED_DIM );
    if(dim_type == einsum_ir::M || dim_type == einsum_ir::N){

      if(l_id_sfc == -1){
        l_id_sfc = l_di;
        (*m_loop_ids)[l_di] = 200000;
      }
      else{
        m_loop_ids->erase(m_loop_ids->begin() + l_di);
        l_di--;
      }
    }
    if(dim_type == einsum_ir::M){
      l_sfc_dims_m.push_back(l_dim_id);
      l_size_sfc_m *= l_dim_size;
    }
    if(dim_type == einsum_ir::N){
      l_sfc_dims_n.push_back(l_dim_id);
      l_size_sfc_n *= l_dim_size; 
    }
  }
  l_size_sfc = l_size_sfc_m * l_size_sfc_n;
  
  m_num_loops = m_loop_ids->size();
  std::cout << "test" << m_loop_ids->size() << std::endl;

  //create vector of offsets
  std::vector< int64_t > l_offsets_m_out;
  std::vector< int64_t > l_offsets_n_out;
  std::vector< int64_t > l_offsets_m_in;
  std::vector< int64_t > l_offsets_n_in;
  l_offsets_m_out.resize( 2 * l_sfc_dims_m.size() + 1);
  l_offsets_n_out.resize( 2 * l_sfc_dims_n.size() + 1);
  l_offsets_m_in.resize(  2 * l_sfc_dims_m.size() + 1);
  l_offsets_n_in.resize(  2 * l_sfc_dims_n.size() + 1);

  convertStridesToOffsets(m_sizes, m_strides_out,   &l_sfc_dims_m, l_offsets_m_out);
  convertStridesToOffsets(m_sizes, m_strides_out,   &l_sfc_dims_n, l_offsets_n_out);
  convertStridesToOffsets(m_sizes, m_strides_left,  &l_sfc_dims_m, l_offsets_m_in);
  convertStridesToOffsets(m_sizes, m_strides_right, &l_sfc_dims_n, l_offsets_n_in);

  // create a vector of sfc movemenets
  std::vector< unsigned char > l_sfc_movement_m;
  std::vector< unsigned char > l_sfc_movement_n;
  l_sfc_movement_m.reserve( l_size_sfc_m * l_size_sfc_n );
  l_sfc_movement_n.reserve( l_size_sfc_m * l_size_sfc_n );
  int64_t l_last_id_m = 0;
  int64_t l_last_id_n = 0;
  for( int64_t l_id = 0; l_id < l_size_sfc_m * l_size_sfc_n; l_id++ ){
    
    //calculate moves in sfc
    int l_id_m, l_id_n;
    gilbert_d2xy( &l_id_m, &l_id_n, l_id, l_size_sfc_m, l_size_sfc_n );

    //map movements to changes in dimensions
    int64_t l_id_jump_m = getMaxDimJump(m_sizes, &l_sfc_dims_m, l_id_m, l_last_id_m);
    int64_t l_id_jump_n = getMaxDimJump(m_sizes, &l_sfc_dims_n, l_id_n, l_last_id_n);

    l_sfc_movement_m[l_id] = l_id_jump_m;
    l_sfc_movement_n[l_id] = l_id_jump_n;

    l_last_id_m = l_id_m;
    l_last_id_n = l_id_n;
  }








#define SIGN(A) (0 < (A) ? (1) : ( 0 == (A) ? (0) : (-1)))

int gilbert_d2xy(int *x, int *y, int idx,int w,int h) {
  *x = 0;
  *y = 0;

  if (w >= h) {
    return gilbert_d2xy_r(idx,0, x,y, w,0, 0,h);
  }
  return gilbert_d2xy_r(idx,0, x,y, 0,h, w,0);
}

int gilbert_d2xy_r(int dst_idx, int cur_idx,
                       int *xres, int *yres,
                       int ax,int ay,
                       int bx,int by ) {
 int nxt_idx;
  int w, h, x, y,
      dax, day,
      dbx, dby,
      di;
  int ax2, ay2, bx2, by2, w2, h2;

  w = abs(ax + ay);
  h = abs(bx + by);

  x = *xres;
  y = *yres;

  /* unit major direction */
  dax = SIGN(ax);
  day = SIGN(ay);

  /* unit orthogonal direction */
  dbx = SIGN(bx);
  dby = SIGN(by);

  di = dst_idx - cur_idx;

  if (h == 1) {
    *xres = x + dax*di;
    *yres = y + day*di;
    return 0;
  }

  if (w == 1) {
    *xres = x + dbx*di;
    *yres = y + dby*di;
    return 0;
  }

  /* floor function */
  ax2 = ax >> 1;
  ay2 = ay >> 1;
  bx2 = bx >> 1;
  by2 = by >> 1;

  w2 = abs(ax2 + ay2);
  h2 = abs(bx2 + by2);

  if ((2*w) > (3*h)) {
    if ((w2 & 1) && (w > 2)) {
      /* prefer even steps */
      ax2 += dax;
      ay2 += day;
    }

    /* long case: split in two parts only */
    nxt_idx = cur_idx + abs((ax2 + ay2)*(bx + by));
    if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
      *xres = x;
      *yres = y;
      return gilbert_d2xy_r(dst_idx, cur_idx,  xres, yres, ax2, ay2, bx, by);
    }
    cur_idx = nxt_idx;

    *xres = x + ax2;
    *yres = y + ay2;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres, yres, ax-ax2, ay-ay2, bx, by);
  }

  if ((h2 & 1) && (h > 2)) {
    /* prefer even steps */
    bx2 += dbx;
    by2 += dby;
  }

  /* standard case: one step up, one long horizontal, one step down */
  nxt_idx = cur_idx + abs((bx2 + by2)*(ax2 + ay2));
  if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
    *xres = x;
    *yres = y;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres,yres, bx2,by2, ax2,ay2);
  }
  cur_idx = nxt_idx;

  nxt_idx = cur_idx + abs((ax + ay)*((bx - bx2) + (by - by2)));
  if ((cur_idx <= dst_idx) && (dst_idx < nxt_idx)) {
    *xres = x + bx2;
    *yres = y + by2;
    return gilbert_d2xy_r(dst_idx, cur_idx, xres,yres, ax,ay, bx-bx2,by-by2);
  }
  cur_idx = nxt_idx;

  *xres = x + (ax - dax) + (bx2 - dbx);
  *yres = y + (ay - day) + (by2 - dby);
  return gilbert_d2xy_r(dst_idx, cur_idx,
                        xres,yres,
                        -bx2, -by2,
                        -(ax-ax2), -(ay-ay2));
}

